import os
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from features.hidden_state_ops import hidden_bank_filename
from registry.model_registry import get_layer_list_and_dim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _encode_single_letter_token(tokenizer: AutoTokenizer, letter: str) -> int:
    """Robustly resolve the single-token id for an MMLU answer letter.

    Different tokenizers (Llama-2, Llama-3.1, Qwen2.5, Gemma, ...) handle the
    BOS prefix differently, so ``tokenizer.encode("A")[1]`` is not portable.
    This helper tries the cheapest correct strategy first and falls back as
    needed.
    """
    # 1) Direct vocabulary lookup, the fast path for BPE / SentencePiece
    #    tokenizers that keep bare letters as their own token.
    token_id = tokenizer.convert_tokens_to_ids(letter)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if token_id is not None and token_id != unk_id:
        return int(token_id)

    # 2) Encode without special tokens; many SentencePiece tokenizers will
    #    emit a leading whitespace token, so prefer the last id which is the
    #    actual letter piece.
    ids = tokenizer.encode(letter, add_special_tokens=False)
    if len(ids) >= 1:
        return int(ids[-1])

    # 3) Last resort: full encode and skip a possible BOS at index 0.
    ids = tokenizer.encode(letter)
    if len(ids) >= 2:
        return int(ids[1])
    return int(ids[0])


class MMLUGenerator:
    """Per-option hidden-state extractor for MMLU prompts.

    The generator is registry-driven: ``layer_list`` may contain any number of
    layer indices, and the produced tensor is always shaped

        ``(num_options=4, num_layers=len(layer_list), hidden_size=layer_dim)``

    so it slots directly into the multi-layer hidden bank
    (``features.hidden_state_ops.hidden_bank_filename`` /
    ``mduq_hidden_bank_dir``).
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        layer_list: List[int],
        layer_dim: int = 4096,
    ):
        # model config
        self.tokenizer = tokenizer
        self.model = model
        # self.max_seq_len = model.config.max_sequence_length
        self.max_seq_len = 2  # 2048
        self.layer_list = list(layer_list)
        self.num_dim = layer_dim
        self.pad_id = model.config.pad_token_id

        if self.pad_id is None:
            self.pad_id = tokenizer.pad_token_id
            if self.pad_id is None:
                self.pad_id = 0

        # Pre-compute the option-letter token ids once; this is both faster
        # and avoids re-running brittle tokenizer logic per prompt.
        self.option_letters: List[str] = ["A", "B", "C", "D"]
        self.letter_token_ids: List[int] = [
            _encode_single_letter_token(tokenizer, letter)
            for letter in self.option_letters
        ]

    @classmethod
    def from_model_name(
        cls,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        model_name: str,
    ) -> "MMLUGenerator":
        """Build a generator using layer/hidden-size info from the registry."""
        layer_list, layer_dim = get_layer_list_and_dim(model_name)
        return cls(model, tokenizer, layer_list, layer_dim)

    def generate_single(
        self,
        prompt_tokens: list,
    ) -> torch.Tensor:
        """Generate per-option last-token hidden states for one MMLU prompt.

        Returns a tensor of shape
        ``(num_options=4, num_layers=len(self.layer_list), hidden_size)``
        that can be either:

        * stacked along a new leading axis and saved per-layer with the
          legacy filename ``"<layer_idx>_output_answer_X.pt"`` (see
          ``supervised_generation.generate_answer_X_mmlu``), or
        * passed to :meth:`save_options_to_bank` to be written under the
          MDUQ multi-layer hidden bank using the canonical
          ``hidden_bank_filename("answer", layer_idx, "last_1_token")``
          naming.
        """
        bsz = 1
        letter_tokens = self.letter_token_ids
        output_last_token_tensor = torch.zeros(
            (len(letter_tokens), len(self.layer_list), self.num_dim),
            dtype=torch.float16,
            device=device,
        )

        prompt_len = len(prompt_tokens)
        if prompt_len == 1:
            prompt_len = len(prompt_tokens[0])

        for k in range(len(letter_tokens)):
            tokens = (
                torch.full((bsz, prompt_len + 1), self.pad_id).to(device).long()
            )
            tokens[0, :prompt_len] = torch.tensor(prompt_tokens).long()
            tokens[0, prompt_len] = letter_tokens[k]
            outputs = self.model.forward(
                tokens[:, : prompt_len + 1],
                use_cache=False,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states
            for idx, layer_idx in enumerate(self.layer_list):
                output_last_token_tensor[k, idx, :] = hidden_states[layer_idx][
                    :, -1, :
                ]

        return output_last_token_tensor

    def save_options_to_bank(
        self,
        output_answer_X: torch.Tensor,
        bank_dir: str,
        layer_list: Optional[Sequence[int]] = None,
    ) -> None:
        """Persist a stacked option-hidden tensor into the MDUQ hidden bank.

        ``output_answer_X`` is expected to be the shape produced by
        ``generate_answer_X_mmlu``, i.e.
        ``(num_examples, num_options=4, num_layers, hidden_size)``. One file
        per layer is written using the canonical
        ``hidden_bank_filename("answer", <layer_idx>, "last_1_token")``
        naming so the bank stays consistent with the QA / WMT pipelines.
        """
        if layer_list is None:
            layer_list = self.layer_list
        if output_answer_X.dim() != 4:
            raise ValueError(
                "output_answer_X must have shape "
                "(num_examples, num_options, num_layers, hidden_size); "
                f"got {tuple(output_answer_X.shape)}"
            )
        if output_answer_X.shape[2] != len(layer_list):
            raise ValueError(
                "output_answer_X.shape[2] does not match len(layer_list): "
                f"{output_answer_X.shape[2]} vs {len(layer_list)}"
            )
        os.makedirs(bank_dir, exist_ok=True)
        for idx, layer_idx in enumerate(layer_list):
            torch.save(
                output_answer_X[:, :, idx, :].contiguous(),
                os.path.join(
                    bank_dir,
                    hidden_bank_filename("answer", layer_idx, "last_1_token"),
                ),
            )
