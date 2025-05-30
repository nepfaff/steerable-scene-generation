"""
Script for uploading a local Hugging Face dataset to the Hugging Face Hub.
"""

import argparse

from steerable_scene_generation.utils.hf_dataset import (
    load_hf_dataset_with_metadata,
    upload_dataset_to_hub,
)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the local dataset folder.",
    )
    parser.add_argument(
        "hub_dataset_id",
        type=str,
        help="The Hugging Face Hub dataset ID (e.g., 'username/dataset-name').",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Whether to make the dataset private.",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Hugging Face API token. If not provided, will try to use the token from "
        "the environment.",
    )
    args = parser.parse_args()

    # Load the dataset and metadata.
    hf_dataset, metadata = load_hf_dataset_with_metadata(args.dataset_path)

    # Upload to Hub.
    upload_dataset_to_hub(
        hf_dataset=hf_dataset,
        metadata=metadata,
        hub_dataset_id=args.hub_dataset_id,
        private=args.private,
        token=args.token,
    )


if __name__ == "__main__":
    main()
