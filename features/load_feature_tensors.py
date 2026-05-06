import json
import os
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from common.runtime_paths import get_test_output_dir
from common.artifact_locator import (
    locate_response_cache_artifacts,
    resolve_runtime_dataset_variant,
)
from common.artifact_paths import (
    ask4conf_jsonl_path,
)
from data.formatters import mmlu_formatter, webgpt_formatter
from registry.model_registry import (
    get_candidate_layers,
    get_local_dir_name,
    get_peer_models,
    resolve_model_id,
)

# Canonical artifact root; ``./test_output`` would collide with future
# source-code packages and is therefore never used directly.
_TEST_OUTPUT_ROOT = str(get_test_output_dir())


def load_MMLU_X_Y(phase, model_name, with_entropy=True, MMLU_TASKS="all"):

    task_list = [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    ]

    if MMLU_TASKS == "Group1":
        task_list = task_list[:40]
    elif MMLU_TASKS == "Group2":
        task_list = task_list[40:]
    elif MMLU_TASKS == "all":
        pass
    else:
        raise ValueError("MMLU_TASKS should be 'Group1','Group2' or 'all'")

    result_dir = "test_output/MMLU/" + model_name + "/" + phase + "/"
    other1_model_name, other2_model_name = get_peer_models(model_name)
    other1_layer_list = get_candidate_layers(other1_model_name)
    other2_layer_list = get_candidate_layers(other2_model_name)
    # Friendly names used in downstream feature columns; preserved for
    # backward compatibility with existing analysis notebooks.
    _peer_friendly_name = {
        "llama_2_7b": "other-7B-",
        "llama_2_13b": "other-13B-",
        "gemma_7b": "other-7B-",
        "gemma_2b": "other-2B-",
    }
    other1_name = _peer_friendly_name.get(other1_model_name, "other1-")
    other2_name = _peer_friendly_name.get(other2_model_name, "other2-")

    other1_result_dir = (
        "test_output/MMLU/" + other1_model_name + "/" + phase + "/"
    )
    other2_result_dir = (
        "test_output/MMLU/" + other2_model_name + "/" + phase + "/"
    )

    other1_exist = True
    if not os.path.exists(other1_result_dir):
        other1_exist = False
    other2_exist = True
    if not os.path.exists(other2_result_dir):
        other2_exist = False

    layer_list = get_candidate_layers(model_name)
    tokenizer = AutoTokenizer.from_pretrained(resolve_model_id(model_name)[0])
    data_total = mmlu_formatter(
        tokenizer=tokenizer,
        num_example=5,
        merge_split=False,
        conv_generation=True,
    )

    for task_idx, task in enumerate(task_list):
        target_dir = result_dir + task + "/"
        other1_target_dir = other1_result_dir + task + "/"
        other2_target_dir = other2_result_dir + task + "/"

        logits = torch.load(target_dir + "query_logits.pt")
        if other1_exist:
            other1_logits = torch.load(other1_target_dir + "query_logits.pt")
        if other2_exist:
            other2_logits = torch.load(other2_target_dir + "query_logits.pt")

        argmax_idx = torch.argmax(logits, dim=1)
        answer_strs = data_total["mmlu__" + task + "__" + phase]["answer_str"]

        # map 'A' 'B' 'C' 'D' to 0 1 2 3
        answer_idx = [
            (
                0
                if answer_strs[i] == "A"
                else (
                    1
                    if answer_strs[i] == "B"
                    else 2 if answer_strs[i] == "C" else 3
                )
            )
            for i in range(len(answer_strs))
        ]
        answer_idx = torch.tensor(answer_idx)
        Y_new = answer_idx == argmax_idx

        def get_file_name_list(layer_list):
            query_average_mid_layer_name = (
                "query_average_layer_" + str(layer_list[0]) + ".pt"
            )
            query_average_last_layer_name = (
                "query_average_layer_" + str(layer_list[1]) + ".pt"
            )
            query_last1_token_mid_layer_name = (
                "query_last_1_token_layer_" + str(layer_list[0]) + ".pt"
            )
            query_last1_token_last_layer_name = (
                "query_last_1_token_layer_" + str(layer_list[1]) + ".pt"
            )
            answerm_mid_layer_name = str(layer_list[0]) + "_output_answer_X.pt"
            answerm_last_layer_name = str(layer_list[1]) + "_output_answer_X.pt"
            return (
                query_average_mid_layer_name,
                query_average_last_layer_name,
                query_last1_token_mid_layer_name,
                query_last1_token_last_layer_name,
                answerm_mid_layer_name,
                answerm_last_layer_name,
            )

        file_name_list = get_file_name_list(layer_list)
        if other1_exist:
            other1_file_name_list = get_file_name_list(other1_layer_list)
        if other2_exist:
            other2_file_name_list = get_file_name_list(other2_layer_list)

        def get_new_X(dir, file_names, argmax_idx):
            data_list = []
            for file_name in file_names:
                data = torch.load(dir + file_name)
                if file_name.startswith("query_last_1_token"):
                    data = data.squeeze()
                    if len(data.shape) == 1:
                        data = data.unsqueeze(0)
                elif file_name.endswith("answer_X.pt"):
                    data_new = torch.stack(
                        [
                            data[i, argmax_idx[i], :]
                            for i in range(len(argmax_idx))
                        ]
                    )
                    data = data_new

                data_list.append(data)

            return data_list

        (
            query_average_mid_layer_new,
            query_average_last_layer_new,
            query_last1_token_mid_layer_new,
            query_last1_token_last_layer_new,
            answerm_mid_layer_new,
            answerm_last_layer_new,
        ) = get_new_X(target_dir, file_name_list, argmax_idx)

        if other1_exist:
            (
                other1_query_average_mid_layer_new,
                other1_query_average_last_layer_new,
                other1_query_last1_token_mid_layer_new,
                other1_query_last1_token_last_layer_new,
                other1_answerm_mid_layer_new,
                other1_answerm_last_layer_new,
            ) = get_new_X(other1_target_dir, other1_file_name_list, argmax_idx)
        if other2_exist:
            (
                other2_query_average_mid_layer_new,
                other2_query_average_last_layer_new,
                other2_query_last1_token_mid_layer_new,
                other2_query_last1_token_last_layer_new,
                other2_answerm_mid_layer_new,
                other2_answerm_last_layer_new,
            ) = get_new_X(other2_target_dir, other2_file_name_list, argmax_idx)

        # get the corresponding argmax_idx along dim=1

        query_answer_ave_mid_new = torch.cat(
            (query_average_mid_layer_new, answerm_mid_layer_new), dim=1
        )
        query_answer_ave_last_new = torch.cat(
            (query_average_last_layer_new, answerm_last_layer_new), dim=1
        )
        query_answer_last_token_mid_new = torch.cat(
            (query_last1_token_mid_layer_new, answerm_mid_layer_new), dim=1
        )
        query_answer_last_token_last_new = torch.cat(
            (query_last1_token_last_layer_new, answerm_last_layer_new), dim=1
        )
        probs_new = torch.nn.Softmax(dim=1)(logits)
        entropy_new = -torch.sum(
            probs_new * torch.log(probs_new), dim=1
        ).reshape(-1, 1)
        sorted_probs = torch.sort(probs_new, dim=1, descending=True).values
        entropy_features = torch.cat((entropy_new, sorted_probs), dim=1)

        if other1_exist:
            other1_query_answer_ave_mid_new = torch.cat(
                (
                    other1_query_average_mid_layer_new,
                    other1_answerm_mid_layer_new,
                ),
                dim=1,
            )
            other1_query_answer_ave_last_new = torch.cat(
                (
                    other1_query_average_last_layer_new,
                    other1_answerm_last_layer_new,
                ),
                dim=1,
            )
            other1_query_answer_last_token_mid_new = torch.cat(
                (
                    other1_query_last1_token_mid_layer_new,
                    other1_answerm_mid_layer_new,
                ),
                dim=1,
            )
            other1_query_answer_last_token_last_new = torch.cat(
                (
                    other1_query_last1_token_last_layer_new,
                    other1_answerm_last_layer_new,
                ),
                dim=1,
            )
            other1_probs_new = torch.nn.Softmax(dim=1)(other1_logits)
            other1_entropy_new = -torch.sum(
                other1_probs_new * torch.log(other1_probs_new), dim=1
            ).reshape(-1, 1)
            other1_sorted_probs = torch.sort(
                other1_probs_new, dim=1, descending=True
            ).values
            other1_entropy_features = torch.cat(
                (other1_entropy_new, other1_sorted_probs), dim=1
            )

        if other2_exist:
            other2_query_answer_ave_mid_new = torch.cat(
                (
                    other2_query_average_mid_layer_new,
                    other2_answerm_mid_layer_new,
                ),
                dim=1,
            )
            other2_query_answer_ave_last_new = torch.cat(
                (
                    other2_query_average_last_layer_new,
                    other2_answerm_last_layer_new,
                ),
                dim=1,
            )
            other2_query_answer_last_token_mid_new = torch.cat(
                (
                    other2_query_last1_token_mid_layer_new,
                    other2_answerm_mid_layer_new,
                ),
                dim=1,
            )
            other2_query_answer_last_token_last_new = torch.cat(
                (
                    other2_query_last1_token_last_layer_new,
                    other2_answerm_last_layer_new,
                ),
                dim=1,
            )
            other2_probs_new = torch.nn.Softmax(dim=1)(other2_logits)
            other2_entropy_new = -torch.sum(
                other2_probs_new * torch.log(other2_probs_new), dim=1
            ).reshape(-1, 1)
            other2_sorted_probs = torch.sort(
                other2_probs_new, dim=1, descending=True
            ).values
            other2_entropy_features = torch.cat(
                (other2_entropy_new, other2_sorted_probs), dim=1
            )

        if with_entropy:
            query_average_mid_layer_new = torch.cat(
                (query_average_mid_layer_new, entropy_features), dim=1
            )
            query_average_last_layer_new = torch.cat(
                (query_average_last_layer_new, entropy_features), dim=1
            )
            query_last1_token_mid_layer_new = torch.cat(
                (query_last1_token_mid_layer_new, entropy_features), dim=1
            )
            query_last1_token_last_layer_new = torch.cat(
                (query_last1_token_last_layer_new, entropy_features), dim=1
            )
            answerm_mid_layer_new = torch.cat(
                (answerm_mid_layer_new, entropy_features), dim=1
            )
            answerm_last_layer_new = torch.cat(
                (answerm_last_layer_new, entropy_features), dim=1
            )
            query_answer_ave_mid_new = torch.cat(
                (query_answer_ave_mid_new, entropy_features), dim=1
            )
            query_answer_ave_last_new = torch.cat(
                (query_answer_ave_last_new, entropy_features), dim=1
            )
            query_answer_last_token_mid_new = torch.cat(
                (query_answer_last_token_mid_new, entropy_features), dim=1
            )
            query_answer_last_token_last_new = torch.cat(
                (query_answer_last_token_last_new, entropy_features), dim=1
            )

            if other1_exist:
                other1_query_average_mid_layer_new = torch.cat(
                    (
                        other1_query_average_mid_layer_new,
                        other1_entropy_features,
                    ),
                    dim=1,
                )
                other1_query_average_last_layer_new = torch.cat(
                    (
                        other1_query_average_last_layer_new,
                        other1_entropy_features,
                    ),
                    dim=1,
                )
                other1_query_last1_token_mid_layer_new = torch.cat(
                    (
                        other1_query_last1_token_mid_layer_new,
                        other1_entropy_features,
                    ),
                    dim=1,
                )
                other1_query_last1_token_last_layer_new = torch.cat(
                    (
                        other1_query_last1_token_last_layer_new,
                        other1_entropy_features,
                    ),
                    dim=1,
                )
                other1_answerm_mid_layer_new = torch.cat(
                    (other1_answerm_mid_layer_new, other1_entropy_features),
                    dim=1,
                )
                other1_answerm_last_layer_new = torch.cat(
                    (other1_answerm_last_layer_new, other1_entropy_features),
                    dim=1,
                )
                other1_query_answer_ave_mid_new = torch.cat(
                    (other1_query_answer_ave_mid_new, other1_entropy_features),
                    dim=1,
                )
                other1_query_answer_ave_last_new = torch.cat(
                    (other1_query_answer_ave_last_new, other1_entropy_features),
                    dim=1,
                )
                other1_query_answer_last_token_mid_new = torch.cat(
                    (
                        other1_query_answer_last_token_mid_new,
                        other1_entropy_features,
                    ),
                    dim=1,
                )
                other1_query_answer_last_token_last_new = torch.cat(
                    (
                        other1_query_answer_last_token_last_new,
                        other1_entropy_features,
                    ),
                    dim=1,
                )

            if other2_exist:
                other2_query_average_mid_layer_new = torch.cat(
                    (
                        other2_query_average_mid_layer_new,
                        other2_entropy_features,
                    ),
                    dim=1,
                )
                other2_query_average_last_layer_new = torch.cat(
                    (
                        other2_query_average_last_layer_new,
                        other2_entropy_features,
                    ),
                    dim=1,
                )
                other2_query_last1_token_mid_layer_new = torch.cat(
                    (
                        other2_query_last1_token_mid_layer_new,
                        other2_entropy_features,
                    ),
                    dim=1,
                )
                other2_query_last1_token_last_layer_new = torch.cat(
                    (
                        other2_query_last1_token_last_layer_new,
                        other2_entropy_features,
                    ),
                    dim=1,
                )
                other2_answerm_mid_layer_new = torch.cat(
                    (other2_answerm_mid_layer_new, other2_entropy_features),
                    dim=1,
                )
                other2_answerm_last_layer_new = torch.cat(
                    (other2_answerm_last_layer_new, other2_entropy_features),
                    dim=1,
                )
                other2_query_answer_ave_mid_new = torch.cat(
                    (other2_query_answer_ave_mid_new, other2_entropy_features),
                    dim=1,
                )
                other2_query_answer_ave_last_new = torch.cat(
                    (other2_query_answer_ave_last_new, other2_entropy_features),
                    dim=1,
                )
                other2_query_answer_last_token_mid_new = torch.cat(
                    (
                        other2_query_answer_last_token_mid_new,
                        other2_entropy_features,
                    ),
                    dim=1,
                )
                other2_query_answer_last_token_last_new = torch.cat(
                    (
                        other2_query_answer_last_token_last_new,
                        other2_entropy_features,
                    ),
                    dim=1,
                )

        if task_idx == 0:
            query_average_mid_layer = query_average_mid_layer_new
            query_average_last_layer = query_average_last_layer_new
            query_last1_token_mid_layer = query_last1_token_mid_layer_new
            query_last1_token_last_layer = query_last1_token_last_layer_new
            answerm_mid_layer = answerm_mid_layer_new
            answerm_last_layer = answerm_last_layer_new
            query_answer_ave_mid = query_answer_ave_mid_new
            query_answer_ave_last = query_answer_ave_last_new
            query_answer_last_token_mid = query_answer_last_token_mid_new
            query_answer_last_token_last = query_answer_last_token_last_new

            if other1_exist:
                other1_query_average_mid_layer = (
                    other1_query_average_mid_layer_new
                )
                other1_query_average_last_layer = (
                    other1_query_average_last_layer_new
                )
                other1_query_last1_token_mid_layer = (
                    other1_query_last1_token_mid_layer_new
                )
                other1_query_last1_token_last_layer = (
                    other1_query_last1_token_last_layer_new
                )
                other1_answerm_mid_layer = other1_answerm_mid_layer_new
                other1_answerm_last_layer = other1_answerm_last_layer_new
                other1_query_answer_ave_mid = other1_query_answer_ave_mid_new
                other1_query_answer_ave_last = other1_query_answer_ave_last_new
                other1_query_answer_last_token_mid = (
                    other1_query_answer_last_token_mid_new
                )
                other1_query_answer_last_token_last = (
                    other1_query_answer_last_token_last_new
                )

            if other2_exist:
                other2_query_average_mid_layer = (
                    other2_query_average_mid_layer_new
                )
                other2_query_average_last_layer = (
                    other2_query_average_last_layer_new
                )
                other2_query_last1_token_mid_layer = (
                    other2_query_last1_token_mid_layer_new
                )
                other2_query_last1_token_last_layer = (
                    other2_query_last1_token_last_layer_new
                )
                other2_answerm_mid_layer = other2_answerm_mid_layer_new
                other2_answerm_last_layer = other2_answerm_last_layer_new
                other2_query_answer_ave_mid = other2_query_answer_ave_mid_new
                other2_query_answer_ave_last = other2_query_answer_ave_last_new
                other2_query_answer_last_token_mid = (
                    other2_query_answer_last_token_mid_new
                )
                other2_query_answer_last_token_last = (
                    other2_query_answer_last_token_last_new
                )

            Y = Y_new

        else:
            query_average_mid_layer = torch.cat(
                (query_average_mid_layer, query_average_mid_layer_new), dim=0
            )
            query_average_last_layer = torch.cat(
                (query_average_last_layer, query_average_last_layer_new), dim=0
            )
            query_last1_token_mid_layer = torch.cat(
                (query_last1_token_mid_layer, query_last1_token_mid_layer_new),
                dim=0,
            )
            query_last1_token_last_layer = torch.cat(
                (
                    query_last1_token_last_layer,
                    query_last1_token_last_layer_new,
                ),
                dim=0,
            )
            answerm_mid_layer = torch.cat(
                (answerm_mid_layer, answerm_mid_layer_new), dim=0
            )
            answerm_last_layer = torch.cat(
                (answerm_last_layer, answerm_last_layer_new), dim=0
            )
            query_answer_ave_mid = torch.cat(
                (query_answer_ave_mid, query_answer_ave_mid_new), dim=0
            )
            query_answer_ave_last = torch.cat(
                (query_answer_ave_last, query_answer_ave_last_new), dim=0
            )
            query_answer_last_token_mid = torch.cat(
                (query_answer_last_token_mid, query_answer_last_token_mid_new),
                dim=0,
            )
            query_answer_last_token_last = torch.cat(
                (
                    query_answer_last_token_last,
                    query_answer_last_token_last_new,
                ),
                dim=0,
            )

            if other1_exist:
                other1_query_average_mid_layer = torch.cat(
                    (
                        other1_query_average_mid_layer,
                        other1_query_average_mid_layer_new,
                    ),
                    dim=0,
                )
                other1_query_average_last_layer = torch.cat(
                    (
                        other1_query_average_last_layer,
                        other1_query_average_last_layer_new,
                    ),
                    dim=0,
                )
                other1_query_last1_token_mid_layer = torch.cat(
                    (
                        other1_query_last1_token_mid_layer,
                        other1_query_last1_token_mid_layer_new,
                    ),
                    dim=0,
                )
                other1_query_last1_token_last_layer = torch.cat(
                    (
                        other1_query_last1_token_last_layer,
                        other1_query_last1_token_last_layer_new,
                    ),
                    dim=0,
                )
                other1_answerm_mid_layer = torch.cat(
                    (other1_answerm_mid_layer, other1_answerm_mid_layer_new),
                    dim=0,
                )
                other1_answerm_last_layer = torch.cat(
                    (other1_answerm_last_layer, other1_answerm_last_layer_new),
                    dim=0,
                )
                other1_query_answer_ave_mid = torch.cat(
                    (
                        other1_query_answer_ave_mid,
                        other1_query_answer_ave_mid_new,
                    ),
                    dim=0,
                )
                other1_query_answer_ave_last = torch.cat(
                    (
                        other1_query_answer_ave_last,
                        other1_query_answer_ave_last_new,
                    ),
                    dim=0,
                )
                other1_query_answer_last_token_mid = torch.cat(
                    (
                        other1_query_answer_last_token_mid,
                        other1_query_answer_last_token_mid_new,
                    ),
                    dim=0,
                )
                other1_query_answer_last_token_last = torch.cat(
                    (
                        other1_query_answer_last_token_last,
                        other1_query_answer_last_token_last_new,
                    ),
                    dim=0,
                )

            if other2_exist:
                other2_query_average_mid_layer = torch.cat(
                    (
                        other2_query_average_mid_layer,
                        other2_query_average_mid_layer_new,
                    ),
                    dim=0,
                )
                other2_query_average_last_layer = torch.cat(
                    (
                        other2_query_average_last_layer,
                        other2_query_average_last_layer_new,
                    ),
                    dim=0,
                )
                other2_query_last1_token_mid_layer = torch.cat(
                    (
                        other2_query_last1_token_mid_layer,
                        other2_query_last1_token_mid_layer_new,
                    ),
                    dim=0,
                )
                other2_query_last1_token_last_layer = torch.cat(
                    (
                        other2_query_last1_token_last_layer,
                        other2_query_last1_token_last_layer_new,
                    ),
                    dim=0,
                )
                other2_answerm_mid_layer = torch.cat(
                    (other2_answerm_mid_layer, other2_answerm_mid_layer_new),
                    dim=0,
                )
                other2_answerm_last_layer = torch.cat(
                    (other2_answerm_last_layer, other2_answerm_last_layer_new),
                    dim=0,
                )
                other2_query_answer_ave_mid = torch.cat(
                    (
                        other2_query_answer_ave_mid,
                        other2_query_answer_ave_mid_new,
                    ),
                    dim=0,
                )
                other2_query_answer_ave_last = torch.cat(
                    (
                        other2_query_answer_ave_last,
                        other2_query_answer_ave_last_new,
                    ),
                    dim=0,
                )
                other2_query_answer_last_token_mid = torch.cat(
                    (
                        other2_query_answer_last_token_mid,
                        other2_query_answer_last_token_mid_new,
                    ),
                    dim=0,
                )
                other2_query_answer_last_token_last = torch.cat(
                    (
                        other2_query_answer_last_token_last,
                        other2_query_answer_last_token_last_new,
                    ),
                    dim=0,
                )

            Y = torch.cat((Y, Y_new), dim=0)

    origin_name_list = [
        "query-ave-mid-layer",
        "query-ave-last-layer",
        "query-last-token-mid-layer",
        "query-last-token-last-layer",
        "answerm-mid-layer",
        "answerm-last-layer",
    ]

    other1_name_list = [other1_name + name for name in origin_name_list]
    other2_name_list = [other2_name + name for name in origin_name_list]
    name_list = origin_name_list + other1_name_list + other2_name_list

    data_list = []
    data_list.append(query_average_mid_layer)
    data_list.append(query_average_last_layer)
    data_list.append(query_last1_token_mid_layer)
    data_list.append(query_last1_token_last_layer)
    data_list.append(answerm_mid_layer)
    data_list.append(answerm_last_layer)

    if other1_exist:
        data_list.append(other1_query_average_mid_layer)
        data_list.append(other1_query_average_last_layer)
        data_list.append(other1_query_last1_token_mid_layer)
        data_list.append(other1_query_last1_token_last_layer)
        data_list.append(other1_answerm_mid_layer)
        data_list.append(other1_answerm_last_layer)

    if other2_exist:
        data_list.append(other2_query_average_mid_layer)
        data_list.append(other2_query_average_last_layer)
        data_list.append(other2_query_last1_token_mid_layer)
        data_list.append(other2_query_last1_token_last_layer)
        data_list.append(other2_answerm_mid_layer)
        data_list.append(other2_answerm_last_layer)

    name_list.append("entropy")
    data_list.append(query_average_mid_layer[:, -5].reshape(-1, 1))

    name_list.append("max prob")
    data_list.append(query_average_mid_layer[:, -4].reshape(-1, 1))

    name_list.append("entropy-supervised")
    data_list.append(query_average_mid_layer[:, -5:])

    name_list.append("query-ans-ave-mid")
    data_list.append(query_answer_ave_mid)

    name_list.append("query-ans-ave-last")
    data_list.append(query_answer_ave_last)

    name_list.append("query-ans-last-token-mid")
    data_list.append(query_answer_last_token_mid)

    name_list.append("query-ans-last-token-last")
    data_list.append(query_answer_last_token_last)

    if other1_exist:
        name_list.append(other1_name + "entropy")
        data_list.append(other1_query_average_mid_layer[:, -5].reshape(-1, 1))

        name_list.append(other1_name + "max prob")
        data_list.append(other1_query_average_mid_layer[:, -4].reshape(-1, 1))

        name_list.append(other1_name + "entropy-supervised")
        data_list.append(other1_query_average_mid_layer[:, -5:])

        name_list.append(other1_name + "query-ans-ave-mid")
        data_list.append(other1_query_answer_ave_mid)

        name_list.append(other1_name + "query-ans-ave-last")
        data_list.append(other1_query_answer_ave_last)

        name_list.append(other1_name + "query-ans-last-token-mid")
        data_list.append(other1_query_answer_last_token_mid)

        name_list.append(other1_name + "query-ans-last-token-last")
        data_list.append(other1_query_answer_last_token_last)

    if other2_exist:
        name_list.append(other2_name + "entropy")
        data_list.append(other2_query_average_mid_layer[:, -5].reshape(-1, 1))

        name_list.append(other2_name + "max prob")
        data_list.append(other2_query_average_mid_layer[:, -4].reshape(-1, 1))

        name_list.append(other2_name + "entropy-supervised")
        data_list.append(other2_query_average_mid_layer[:, -5:])

        name_list.append(other2_name + "query-ans-ave-mid")
        data_list.append(other2_query_answer_ave_mid)

        name_list.append(other2_name + "query-ans-ave-last")
        data_list.append(other2_query_answer_ave_last)

        name_list.append(other2_name + "query-ans-last-token-mid")
        data_list.append(other2_query_answer_last_token_mid)

        name_list.append(other2_name + "query-ans-last-token-last")
        data_list.append(other2_query_answer_last_token_last)

    if phase == "validation":
        try:
            if model_name == "llama_3_8b":
                filename = "test_output/ask4conf/llama_3_8b/mmlu.json"
                with open(filename) as f:
                    ask4conf_score = json.load(f)
                ask4conf_score = ask4conf_score["ask4conf_prob"]
                ask4conf_score = list(ask4conf_score.values())
                if MMLU_TASKS == "Group1":
                    ask4conf_score = ask4conf_score[: data_list[-1].shape[0]]
                elif MMLU_TASKS == "Group2":
                    ask4conf_score = ask4conf_score[-data_list[-1].shape[0] :]
            else:
                ask4conf_score = load_ask4conf_score(
                    dataset_name="MMLU",
                    model_type=model_name,
                    available_idxs=list(
                        range(query_answer_last_token_last.shape[0])
                    ),
                    MMLU_TASKS=MMLU_TASKS,
                )
        except:
            ask4conf_score = None
    else:
        ask4conf_score = None

    return data_list, name_list, Y, ask4conf_score


def get_index_of_valid_X(dataset_name, model_type, phase="test"):

    SU_KEY = "semantic_entropy"
    dataset_name = dataset_name

    if dataset_name.startswith("wmt"):
        metric = "bleu"
    else:
        metric = "rouge1_most"

    artifacts = locate_response_cache_artifacts(dataset_name, model_type)
    dataset_name = artifacts.dataset_variant
    output_dir = str(artifacts.response_cache_dir) + "/"
    data_json_path = str(artifacts.paths["mextend"])

    if dataset_name.startswith("wmt"):
        mrouge_path = str(artifacts.paths["mextend_bleu"])
    else:
        mrouge_path = str(artifacts.paths["mextend_rouge"])

    if not os.path.exists(data_json_path):
        raise FileNotFoundError(
            f"[get_index_of_valid_X] missing _mextend.json for split_tag="
            f"{dataset_name} model={model_type}: {data_json_path}"
        )
    if not os.path.exists(mrouge_path):
        raise FileNotFoundError(
            f"[get_index_of_valid_X] missing metric file for split_tag="
            f"{dataset_name} model={model_type}: {mrouge_path}"
        )

    with open(data_json_path) as fr:
        data = json.load(fr)

    with open(mrouge_path) as fr:
        mrouge = json.load(fr)

    has_SU = False
    SU_path = str(artifacts.paths["semantic_entropy"])
    if os.path.exists(SU_path):
        has_SU = True

    if phase == "test" and has_SU:
        with open(SU_path) as fr:
            SU_data = json.load(fr)
    else:
        SU_data = data

    available_idxs = []
    mrouges = []
    SU_scores = []

    for d_idx in range(len(data)):
        if phase == "test":
            if d_idx >= len(SU_data):
                break

        if metric in mrouge[d_idx]:
            if phase != "test" or (SU_KEY in SU_data[d_idx]) or (not has_SU):
                available_idxs.append(d_idx)
                mrouges.append(mrouge[d_idx][metric])
                if SU_KEY in SU_data[d_idx]:
                    SU_scores.append(-SU_data[d_idx][SU_KEY])

    if dataset_name.startswith("triviaqa") or dataset_name.startswith("coqa"):
        if phase == "test":
            available_idxs = available_idxs[:2000]
            mrouges = mrouges[:2000]
            if has_SU:
                SU_scores = SU_scores[:2000]
        else:
            available_idxs = available_idxs[2000:]
            mrouges = mrouges[2000:]
    if phase == "test":
        try:
            if model_type == "llama_3_8b":
                output_dir = "test_output/ask4conf/llama_3_8b/"
                if dataset_name.startswith("triviaqa"):
                    filename = output_dir + "triviaqa.json"
                elif dataset_name.startswith("coqa"):
                    filename = output_dir + "coqa.json"
                elif dataset_name.startswith("wmt"):
                    filename = output_dir + "wmt.json"

                with open(filename) as fr:
                    ask4conf_score = json.load(fr)
                ask4conf_score = ask4conf_score["ask4conf_prob"]
                # get the values from the dict ask4conf_score
                ask4conf_score = list(ask4conf_score.values())
                if len(ask4conf_score) > 2000:
                    ask4conf_score = ask4conf_score[:2000]
            else:
                ask4conf_score = load_ask4conf_score(
                    dataset_name=dataset_name,
                    model_type=model_type,
                    available_idxs=available_idxs,
                )
        except:
            ask4conf_score = None
    else:
        ask4conf_score = None

    if not has_SU:
        SU_scores = None

    return available_idxs, mrouges, SU_scores, ask4conf_score


def load_X_Y_with_phase(
    dataset_name, model_type, phase="test", with_entropy=True
):
    dataset_variant = resolve_runtime_dataset_variant(
        dataset_name, model_name=model_type
    )
    output_dir = _TEST_OUTPUT_ROOT + "/" + dataset_variant + "/" + model_type + "/"

    available_idxs, mrouges, SU_scores, ask4conf_score = get_index_of_valid_X(
        dataset_name, model_type, phase=phase
    )
    # prepare X
    layer_list = get_candidate_layers(model_type)
    other1_model_name, other2_model_name = get_peer_models(model_type)
    layer_list_other1 = get_candidate_layers(other1_model_name)
    layer_list_other2 = get_candidate_layers(other2_model_name)
    _peer_friendly_name = {
        "llama_2_7b": "other-7B-",
        "llama_2_13b": "other-13B-",
        "gemma_7b": "other-7B-",
        "gemma_2b": "other-2B-",
    }
    other1_name = _peer_friendly_name.get(other1_model_name, "other1-")
    other2_name = _peer_friendly_name.get(other2_model_name, "other2-")
    other1_variant = resolve_runtime_dataset_variant(
        dataset_name, model_name=other1_model_name
    )
    other2_variant = resolve_runtime_dataset_variant(
        dataset_name, model_name=other2_model_name
    )
    other1_output_dir = (
        _TEST_OUTPUT_ROOT + "/" + other1_variant + "/" + other1_model_name + "/"
    )
    other2_output_dir = (
        _TEST_OUTPUT_ROOT + "/" + other2_variant + "/" + other2_model_name + "/"
    )

    def get_file_name_list(layer_list, is_cross=False):

        if not is_cross:
            query_average_mid_layer_name = (
                "query_average_layer_" + str(layer_list[0]) + ".pt"
            )
            query_average_last_layer_name = (
                "query_average_layer_" + str(layer_list[1]) + ".pt"
            )
            query_last1_token_mid_layer_name = (
                "query_last_1_token_layer_" + str(layer_list[0]) + ".pt"
            )
            query_last1_token_last_layer_name = (
                "query_last_1_token_layer_" + str(layer_list[1]) + ".pt"
            )
            answerm_mid_layer_name = (
                "answer_average_layer_" + str(layer_list[0]) + ".pt"
            )
            answerm_last_layer_name = (
                "answer_average_layer_" + str(layer_list[1]) + ".pt"
            )
            answer_last1_token_mid_layer_name = (
                "answer_last_1_token_layer_" + str(layer_list[0]) + ".pt"
            )
            answer_last1_token_last_layer_name = (
                "answer_last_1_token_layer_" + str(layer_list[1]) + ".pt"
            )
            file_name_list = [
                query_average_mid_layer_name,
                query_average_last_layer_name,
                query_last1_token_mid_layer_name,
                query_last1_token_last_layer_name,
                answerm_mid_layer_name,
                answerm_last_layer_name,
                answer_last1_token_mid_layer_name,
                answer_last1_token_last_layer_name,
            ]
        else:
            query_average_mid_layer_name = (
                "cross_query_average_layer_" + str(layer_list[0]) + ".pt"
            )
            query_average_last_layer_name = (
                "cross_query_average_layer_" + str(layer_list[1]) + ".pt"
            )
            query_last1_token_mid_layer_name = (
                "cross_query_last_1_token_layer_" + str(layer_list[0]) + ".pt"
            )
            query_last1_token_last_layer_name = (
                "cross_query_last_1_token_layer_" + str(layer_list[1]) + ".pt"
            )
            other_answerm_mid_layer_name = (
                "cross_answer_average_layer_" + str(layer_list[0]) + ".pt"
            )
            other_answerm_last_layer_name = (
                "cross_answer_average_layer_" + str(layer_list[1]) + ".pt"
            )
            other_answer_last1_token_mid_layer_name = (
                "cross_answer_last_1_token_layer_" + str(layer_list[0]) + ".pt"
            )
            other_answer_last1_token_last_layer_name = (
                "cross_answer_last_1_token_layer_" + str(layer_list[1]) + ".pt"
            )
            file_name_list = [
                query_average_mid_layer_name,
                query_average_last_layer_name,
                query_last1_token_mid_layer_name,
                query_last1_token_last_layer_name,
                other_answerm_mid_layer_name,
                other_answerm_last_layer_name,
                other_answer_last1_token_mid_layer_name,
                other_answer_last1_token_last_layer_name,
            ]

        return file_name_list

    file_name_list = get_file_name_list(layer_list)
    other1_file_name_list = get_file_name_list(layer_list_other1, is_cross=True)
    other2_file_name_list = get_file_name_list(layer_list_other2, is_cross=True)

    other1_exist = True
    other2_exist = True
    for file_name in other1_file_name_list:
        if not os.path.exists(other1_output_dir + file_name):
            other1_exist = False
            break
    for file_name in other2_file_name_list:
        if not os.path.exists(other2_output_dir + file_name):
            other2_exist = False
            break

    def load_file(file_name_list, output_dir, available_idxs):
        data_list = []
        for file_name in file_name_list:
            data = torch.load(output_dir + file_name)
            data = data[available_idxs].squeeze()
            data_list.append(data)
        return data_list

    X_list = load_file(file_name_list, output_dir, available_idxs)
    if other1_exist:
        other1_X_list = load_file(
            other1_file_name_list, other1_output_dir, available_idxs
        )
    if other2_exist:
        other2_X_list = load_file(
            other2_file_name_list, other2_output_dir, available_idxs
        )

    X_list.append(
        torch.cat((X_list[0], X_list[4]), dim=1)
    )  # query_ans_ave_mid_layer
    X_list.append(
        torch.cat((X_list[1], X_list[5]), dim=1)
    )  # query_ans_ave_last_layer
    X_list.append(
        torch.cat((X_list[2], X_list[6]), dim=1)
    )  # query_ans_last_token_mid_layer
    X_list.append(
        torch.cat((X_list[3], X_list[7]), dim=1)
    )  # query_ans_last_token_last_layer

    if other1_exist:
        other1_X_list.append(
            torch.cat((other1_X_list[0], other1_X_list[4]), dim=1)
        )  # query_ans_ave_mid_layer
        other1_X_list.append(
            torch.cat((other1_X_list[1], other1_X_list[5]), dim=1)
        )  # query_ans_ave_last_layer
        other1_X_list.append(
            torch.cat((other1_X_list[2], other1_X_list[6]), dim=1)
        )  # query_ans_last_token_mid_layer
        other1_X_list.append(
            torch.cat((other1_X_list[3], other1_X_list[7]), dim=1)
        )  # query_ans_last_token_last_layer
    else:
        other1_X_list = []

    if other2_exist:
        other2_X_list.append(
            torch.cat((other2_X_list[0], other2_X_list[4]), dim=1)
        )  # query_ans_ave_mid_layer
        other2_X_list.append(
            torch.cat((other2_X_list[1], other2_X_list[5]), dim=1)
        )  # query_ans_ave_last_layer
        other2_X_list.append(
            torch.cat((other2_X_list[2], other2_X_list[6]), dim=1)
        )  # query_ans_last_token_mid_layer
        other2_X_list.append(
            torch.cat((other2_X_list[3], other2_X_list[7]), dim=1)
        )  # query_ans_last_token_last_layer
    else:
        other2_X_list = []

    name_list = [
        "query-ave mid layer",
        "query-ave last layer",
        "query last-token mid layer",
        "query last-token last layer",
        "answerm-ave mid layer",
        "answerm-ave last layer",
        "answerm last-token mid layer",
        "answerm last-token last layer",
        "query-ans-ave-mid",
        "query-ans-ave-last",
        "query-ans-last-token-mid",
        "query-ans-last-token-last",
    ]
    if other1_exist:
        other1_name_list = [other1_name + name for name in name_list]
    else:
        other1_name_list = []
    if other2_exist:
        other2_name_list = [other2_name + name for name in name_list]
    else:
        other2_name_list = []

    name_list = name_list + other1_name_list + other2_name_list

    data_list = X_list + other1_X_list + other2_X_list

    # load entropy data
    def load_entropy_probs(dir, is_cross=False):
        query_entropy_name = "query_entropies.pt"
        answerm_entropy_name = "answer_entropies.pt"
        query_probs_name = "query_probs.pt"
        answerm_probs_name = "answer_probs.pt"

        if is_cross:
            query_entropy_name = "cross_query_entropies.pt"
            answerm_entropy_name = "cross_answer_entropies.pt"
            query_probs_name = "cross_query_probs.pt"
            answerm_probs_name = "cross_answer_probs.pt"

        query_entropies = torch.load(dir + query_entropy_name)
        answerm_entropies = torch.load(dir + answerm_entropy_name)
        query_probs = torch.load(dir + query_probs_name)
        answerm_probs = torch.load(dir + answerm_probs_name)
        query_entropies = query_entropies[available_idxs]
        answerm_entropies = answerm_entropies[available_idxs]
        query_probs = query_probs[available_idxs]
        answerm_probs = answerm_probs[available_idxs]

        return query_entropies, answerm_entropies, query_probs, answerm_probs

    query_entropies, answerm_entropies, query_probs, answerm_probs = (
        load_entropy_probs(output_dir)
    )

    if other1_exist:
        (
            other1_query_entropies,
            other1_answerm_entropies,
            other1_query_probs,
            other1_answerm_probs,
        ) = load_entropy_probs(other1_output_dir, is_cross=True)

    if other2_exist:
        (
            other2_query_entropies,
            other2_answerm_entropies,
            other2_query_probs,
            other2_answerm_probs,
        ) = load_entropy_probs(other2_output_dir, is_cross=True)

    query_entropies = torch.cat((query_entropies, query_probs), dim=1)
    answerm_entropies = torch.cat((answerm_entropies, answerm_probs), dim=1)

    if other1_exist:
        other1_query_entropies = torch.cat(
            (other1_query_entropies, other1_query_probs), dim=1
        )
        other1_answerm_entropies = torch.cat(
            (other1_answerm_entropies, other1_answerm_probs), dim=1
        )

    if other2_exist:
        other2_query_entropies = torch.cat(
            (other2_query_entropies, other2_query_probs), dim=1
        )
        other2_answerm_entropies = torch.cat(
            (other2_answerm_entropies, other2_answerm_probs), dim=1
        )

    if with_entropy:
        # concatenate extropy_data to all the layer data
        for idx in range(len(data_list)):
            if name_list[idx].startswith("query-ans"):
                data_list[idx] = torch.cat(
                    (data_list[idx], query_entropies, answerm_entropies), dim=1
                )

            elif name_list[idx].startswith("query"):
                data_list[idx] = torch.cat(
                    (data_list[idx], query_entropies), dim=1
                )

            elif name_list[idx].startswith("answerm"):
                data_list[idx] = torch.cat(
                    (data_list[idx], answerm_entropies), dim=1
                )

            elif name_list[idx].startswith(other1_name + "query-ans"):
                data_list[idx] = torch.cat(
                    (
                        data_list[idx],
                        other1_query_entropies,
                        other1_answerm_entropies,
                    ),
                    dim=1,
                )

            elif name_list[idx].startswith(other1_name + "query"):
                data_list[idx] = torch.cat(
                    (data_list[idx], other1_query_entropies), dim=1
                )

            elif name_list[idx].startswith(other1_name + "answerm"):
                data_list[idx] = torch.cat(
                    (data_list[idx], other1_answerm_entropies), dim=1
                )

            elif name_list[idx].startswith(other2_name + "query-ans"):
                data_list[idx] = torch.cat(
                    (
                        data_list[idx],
                        other2_query_entropies,
                        other2_answerm_entropies,
                    ),
                    dim=1,
                )
            elif name_list[idx].startswith(other2_name + "query"):
                data_list[idx] = torch.cat(
                    (data_list[idx], other2_query_entropies), dim=1
                )
            elif name_list[idx].startswith(other2_name + "answerm"):
                data_list[idx] = torch.cat(
                    (data_list[idx], other2_answerm_entropies), dim=1
                )

    y = mrouges

    for column_idx in range(answerm_entropies.shape[1]):
        data_list.append(answerm_entropies[:, column_idx].reshape(-1, 1))
    data_list.append(query_entropies)
    data_list.append(answerm_entropies)
    data_list.append(torch.cat((query_entropies, answerm_entropies), dim=1))

    extend_name_list = [
        "Max Entropy",
        "Min Entropy",
        "Entropy Avg",
        "Entropy Std",
        "Max Prob",
        "Min Prob",
        "Prob Mean",
        "Prob Std",
        "Log Prob Mean",
        "Log Prob Std",
        "supervised-query-entropy",
        "supervised-answer-entropy",
        "supervised-query-answer-entropy",
    ]
    name_list.extend(extend_name_list)

    if other1_exist:
        for column_idx in range(other1_answerm_entropies.shape[1]):
            data_list.append(
                other1_answerm_entropies[:, column_idx].reshape(-1, 1)
            )
        data_list.append(other1_query_entropies)
        data_list.append(other1_answerm_entropies)
        data_list.append(
            torch.cat((other1_query_entropies, other1_answerm_entropies), dim=1)
        )

        name_list.extend([other1_name + name for name in extend_name_list])

    if other2_exist:
        for column_idx in range(other2_answerm_entropies.shape[1]):
            data_list.append(
                other2_answerm_entropies[:, column_idx].reshape(-1, 1)
            )
        data_list.append(other2_query_entropies)
        data_list.append(other2_answerm_entropies)
        data_list.append(
            torch.cat((other2_query_entropies, other2_answerm_entropies), dim=1)
        )

        name_list.extend([other2_name + name for name in extend_name_list])

    return data_list, name_list, y, SU_scores, ask4conf_score


def load_X_Y(dataset_name, model_type, with_entropy=True):

    # load test data
    if dataset_name == "coqa":
        test_dataset_name = "coqa__train"
        train_dataset_name = "coqa__train"
    elif dataset_name == "triviaqa":
        test_dataset_name = "triviaqa__train"
        train_dataset_name = "triviaqa__train"
    elif dataset_name == "wmt":
        test_dataset_name = "wmt__test"
        train_dataset_name = "wmt__train"
    else:
        # raise error to hint the name should be in list['coqa','triviaqa','wmt']
        raise ValueError(
            "dataset name not supported, should be in ['coqa','triviaqa','wmt']"
        )

    # load test data
    data_test_list, name_test_list, y_test, SU_test, ask4conf_test = (
        load_X_Y_with_phase(
            test_dataset_name,
            model_type,
            phase="test",
            with_entropy=with_entropy,
        )
    )

    # load train data
    data_train_list, name_train_list, y_train, _, _ = load_X_Y_with_phase(
        train_dataset_name, model_type, phase="train", with_entropy=with_entropy
    )

    return (
        data_train_list,
        data_test_list,
        name_train_list,
        y_train,
        y_test,
        SU_test,
        ask4conf_test,
    )


from typing import Literal


def load_ask4conf_score(
    dataset_name: Literal[
        "coqa__test", "triviaqa__train", "wmt__test", "MMLU", "coqa__train"
    ],
    model_type: Literal["llama_2_7b", "gemma_7b"],
    available_idxs: list[int],
    MMLU_TASKS: Literal[
        "all",
        "Group1",
        "Group2",
    ] = "all",
) -> pd.Series:
    MMLU_TASK_list = [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    ]

    if MMLU_TASKS == "Group1":
        MMLU_TASKS = MMLU_TASK_list[:40]
    elif MMLU_TASKS == "Group2":
        MMLU_TASKS = MMLU_TASK_list[40:]
    elif MMLU_TASKS == "all":
        MMLU_TASKS = MMLU_TASK_list
    else:
        raise ValueError("MMLU_TASKS should be in ['Group1','Group2','all']")

    # Canonical jsonl filename is ``<split_tag>.jsonl``. Historically the
    # writer produced double-prefixed files like
    # ``triviaqa__triviaqa__train.jsonl`` -- those are migrated/normalized
    # via the canonical helper.
    files = {
        "coqa__test": ["coqa__test.jsonl"],
        "coqa__train": ["coqa__train.jsonl"],
        "triviaqa__train": ["triviaqa__train.jsonl"],
        "triviaqa__test": ["triviaqa__test.jsonl"],
        "wmt__test": ["wmt__test.jsonl"],
        "wmt__train": ["wmt__train.jsonl"],
        "MMLU": [
            f"mmlu__{task}__validation.jsonl" for task in MMLU_TASKS
        ],
    }

    files_to_read = [
        ask4conf_jsonl_path(fname.removesuffix(".jsonl"), model_type)
        for fname in files[dataset_name]
    ]

    dfs = []
    for f in files_to_read:
        try:
            df = pd.read_json(str(f), lines=True)
            dfs.append(df)
        except Exception as e:
            print(f"{f}")

    df = pd.concat(dfs, axis=0).reset_index(drop=True)
    if "sample_idx" in df.columns and df["sample_idx"].is_unique:
        by_sample_idx = df.set_index("sample_idx")["prob"]
        return by_sample_idx.reindex(available_idxs)
    return df["prob"].iloc[
        [_ for _ in available_idxs if _ < df["prob"].index.stop]    # type: ignore
    ]


# ---------------------------------------------------------------------------
# MDUQ multi-layer hidden-bank loaders
# ---------------------------------------------------------------------------
#
# The legacy ``load_X_Y`` / ``load_MMLU_X_Y`` helpers above hard-code two
# layers and the legacy filename scheme. The two functions below are the
# multi-layer entry points used by the new MDUQ pipeline; they read from
#
#     ./test_output/<dataset>/<model>/mduq/hidden_bank/
#
# (see ``features.hidden_state_ops.mduq_hidden_bank_dir``) and stack tensors
# along a new ``L`` axis so any number of candidate layers is supported.

from typing import Any, Dict, List as _List, Optional as _Optional, Sequence as _Sequence  # noqa: E402

from features.hidden_state_ops import (  # noqa: E402
    hidden_bank_filename,
    mduq_hidden_bank_dir,
)
from registry.model_registry import get_candidate_layers as _get_candidate_layers  # noqa: E402


_MDUQ_DEFAULT_VIEWS = ("query", "answer")
_MDUQ_DEFAULT_KINDS = ("average", "last_1_token")
_MDUQ_DEFAULT_EXTRAS = (
    "query_entropies.pt",
    "query_probs.pt",
    "query_entropy_available.pt",
    "query_prob_available.pt",
    "answer_entropy_available.pt",
    "answer_prob_available.pt",
    "entropy_missing_reasons.json",
    "query_logits.pt",
    "labels.pt",
    "Y.pt",
)


def _stack_layers(per_layer: _List[torch.Tensor]) -> torch.Tensor:
    """Stack a list of per-layer tensors along a new axis at position 1.

    Each entry is ``(N, ...)``; result is ``(N, L, ...)``. Tensors are moved
    to CPU and cast to float32 to keep downstream math numerically stable.
    """
    return torch.stack(
        [t.detach().to("cpu").float() for t in per_layer], dim=1
    )


def load_multilayer_feature_bank(
    dataset_name: str,
    model_name: str,
    *,
    layer_list: _Optional[_Sequence[int]] = None,
    output_root: _Optional[str] = None,
    views: _Sequence[str] = _MDUQ_DEFAULT_VIEWS,
    kinds: _Sequence[str] = _MDUQ_DEFAULT_KINDS,
    extras: _Optional[_Sequence[str]] = None,
    bank_dir: _Optional[str] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """Load a model's multi-layer hidden bank from disk.

    Parameters
    ----------
    dataset_name:
        Bank-relative dataset key. For QA / WMT / AmbigQA / TruthfulQA pass
        the dataset name (e.g. ``"triviaqa"``); for MMLU pass
        ``"MMLU/<phase>"`` or ``"MMLU/<phase>/<task>"`` so the same on-disk
        layout used by ``mduq_hidden_bank_dir`` is honoured.
    model_name:
        Registry key of the model whose bank we are loading.
    layer_list:
        Subset of layers to load. When ``None`` we use the model's full
        ``candidate_layers`` from the registry.
    output_root:
        Root of the test-output tree. Defaults to ``./test_output``.
    views, kinds:
        Which view (``"query"``, ``"answer"``) and kind (``"average"``,
        ``"last_1_token"``) tensors to attempt to load.
    extras:
        Iterable of extra filenames (relative to ``bank_dir``) to load into
        ``result["extras"]``. Defaults to a small set of standard side
        files; missing extras are skipped silently unless ``strict``.
    bank_dir:
        Override the bank directory. Defaults to
        ``mduq_hidden_bank_dir(dataset_name, model_name, output_root)``.
    strict:
        When ``True``, raise ``FileNotFoundError`` if any requested
        view/kind/extra file is missing instead of skipping it.

    Returns
    -------
    dict with keys:
        ``"query"``    -> ``{kind: tensor of shape (N, L, ...)}``  (only present kinds)
        ``"answer"``   -> ``{kind: tensor of shape (N, L, ...)}``  (only present kinds)
        ``"extras"``   -> ``{filename_stem: tensor}`` for whatever extras existed
        ``"layer_list"`` -> ``list[int]``
        ``"bank_dir"`` -> ``str``
    """

    if layer_list is None:
        layer_list = _get_candidate_layers(model_name)
    layer_list = list(layer_list)

    if output_root is None:
        from common.runtime_paths import get_test_output_dir
        output_root = str(get_test_output_dir())

    if bank_dir is None:
        bank_dir = mduq_hidden_bank_dir(dataset_name, model_name, output_root=output_root)
        # Fallback to legacy ``mduq/hidden_bank`` when the canonical
        # ``diaguq/hidden_bank`` does not exist yet.
        if not os.path.isdir(bank_dir):
            legacy_bank = os.path.join(
                output_root, dataset_name, model_name, "mduq", "hidden_bank"
            )
            if os.path.isdir(legacy_bank):
                bank_dir = legacy_bank

    if not os.path.isdir(bank_dir):
        raise FileNotFoundError(
            f"DiagUQ hidden bank not found at {bank_dir!r}; "
            "run `python run.py build-hidden-bank ...` first for this "
            f"(dataset='{dataset_name}', model='{model_name}') pair."
        )

    if extras is None:
        extras = _MDUQ_DEFAULT_EXTRAS

    result: Dict[str, Any] = {
        "layer_list": layer_list,
        "bank_dir": bank_dir,
        "extras": {},
        "extra_paths": {},
    }

    for view in views:
        view_dict: Dict[str, torch.Tensor] = {}
        for kind in kinds:
            per_layer: _List[torch.Tensor] = []
            missing = False
            for layer_idx in layer_list:
                fname = hidden_bank_filename(view, layer_idx, kind)
                fpath = os.path.join(bank_dir, fname)
                if not os.path.isfile(fpath):
                    missing = True
                    if strict:
                        raise FileNotFoundError(fpath)
                    break
                per_layer.append(torch.load(fpath, map_location="cpu"))
            if not missing and per_layer:
                view_dict[kind] = _stack_layers(per_layer)
        if view_dict:
            result[view] = view_dict

    for extra_name in extras:
        fpath = os.path.join(bank_dir, extra_name)
        if not os.path.isfile(fpath):
            if strict:
                raise FileNotFoundError(fpath)
            continue
        key = os.path.splitext(extra_name)[0]
        if extra_name.endswith(".json"):
            with open(fpath, "r", encoding="utf-8") as fr:
                result["extras"][key] = json.load(fr)
        else:
            result["extras"][key] = torch.load(fpath, map_location="cpu")
        result["extra_paths"][key] = fpath

    return result


def load_mduq_dataset(
    dataset_name: str,
    model_name: str,
    *,
    layer_list: _Optional[_Sequence[int]] = None,
    output_root: _Optional[str] = None,
    query_kind: str = "average",
    answer_kind: str = "average",
    relation_ops: _Optional[_Sequence[str]] = None,
    option_idx: _Optional[torch.Tensor] = None,
    label_key: str = "labels",
    fallback_label_key: str = "Y",
    bank_dir: _Optional[str] = None,
    require_entropy: bool = False,
    include_entropy_availability: bool = True,
) -> Dict[str, Any]:
    """High-level loader that returns the four MDUQ views plus labels.

    Combines :func:`load_multilayer_feature_bank` with
    :func:`features.build_multiview_features.build_all_views` so a downstream
    trainer can simply do::

        bundle = load_mduq_dataset("triviaqa", "qwen_2_5_7b_instruct")
        X_query   = bundle["views"]["query"]      # (N, L, D)
        X_answer  = bundle["views"]["answer"]     # (N, L, D)
        X_rel     = bundle["views"]["relation"]   # (N, L, F_r)
        X_entropy = bundle["views"].get("entropy") # optional (N, F_e)
        Y         = bundle["labels"]              # (N,) or None

    Returns a dict with keys ``"views"``, ``"labels"``, ``"layer_list"``,
    ``"bank_dir"``, ``"extras"``, ``"bank"`` (raw bank for advanced uses).
    """
    # Local import to avoid a circular dependency at module load time.
    from features.build_multiview_features import (  # noqa: WPS433
        RELATION_OPS_DEFAULT,
        build_all_views,
    )

    bank = load_multilayer_feature_bank(
        dataset_name,
        model_name,
        layer_list=layer_list,
        output_root=output_root,
        bank_dir=bank_dir,
    )

    views = build_all_views(
        bank,
        query_kind=query_kind,
        answer_kind=answer_kind,
        relation_ops=(
            relation_ops if relation_ops is not None else RELATION_OPS_DEFAULT
        ),
        option_idx=option_idx,
        require_entropy=require_entropy,
        include_entropy_availability=include_entropy_availability,
        extra_paths=bank.get("extra_paths", {}),
    )

    labels = None
    extras = bank.get("extras", {})
    if label_key in extras:
        labels = extras[label_key]
    elif fallback_label_key in extras:
        labels = extras[fallback_label_key]

    return {
        "views": views,
        "labels": labels,
        "layer_list": bank["layer_list"],
        "bank_dir": bank["bank_dir"],
        "extras": extras,
        "extra_paths": bank.get("extra_paths", {}),
        "bank": bank,
    }
