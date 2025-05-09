"""
Script for computing the maximum number of language tokens required to encode the
scene language annotations in a dataset.
The number will be added to the dataset metadata.
"""

import argparse
import os

import torch

from omegaconf import DictConfig

from steerable_scene_generation.algorithms.common.txt_encoding import (
    load_txt_encoder_from_config,
)
from steerable_scene_generation.utils.hf_dataset import (
    load_hf_dataset_with_metadata,
    save_hf_dataset_metadata,
)

# Disable tokenizer parallelism.
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the dataset to augment.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=os.cpu_count(),
        help="The maximum number of parallel workers to use.",
    )
    parser.add_argument(
        "--txt_encoder",
        type=str,
        default="bert",
        help="The text encoder to use.",
    )
    parser.add_argument(
        "--txt_encoder_size",
        type=str,
        default="base",
        help="The size of the text encoder to use.",
    )
    args = parser.parse_args()

    # Load the dataset and metadata.
    dataset, metadata = load_hf_dataset_with_metadata(args.dataset_path)

    tokenizer, _ = load_txt_encoder_from_config(
        cfg=DictConfig(
            {
                "classifier_free_guidance": {
                    "max_length": -1,  # Ignored but required for initialization.
                    "txt_encoder": args.txt_encoder,
                    "txt_encoder_size": args.txt_encoder_size,
                }
            }
        ),
        component="tokenizer",
    )

    def compute_num_language_tokens(item):
        prompt = item["language_annotation"]
        num_tokens = tokenizer.get_num_required_tokens(prompt)
        return {"num_language_tokens": num_tokens}

    result = dataset.map(
        compute_num_language_tokens,
        num_proc=args.max_workers,
        batched=False,
        desc="Computing max required language token number",
        remove_columns=dataset.column_names,
    )

    # Get the max number of language tokens.
    largest_index = torch.argmax(result["num_language_tokens"]).item()
    max_num_language_tokens = result["num_language_tokens"][largest_index].item()
    print(f"Max number of language tokens: {max_num_language_tokens}")
    print(f"Longest annotation: {dataset[largest_index]['language_annotation']}")

    # Add the max number of language tokens to the metadata.
    metadata["language_tokens"] = {
        "max_num_tokens": max_num_language_tokens,
        "txt_encoder": args.txt_encoder,
        "txt_encoder_size": args.txt_encoder_size,
    }

    # Update the metadata.
    save_hf_dataset_metadata(metadata=metadata, dataset_path=args.dataset_path)


if __name__ == "__main__":
    main()
