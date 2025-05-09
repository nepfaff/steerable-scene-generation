"""
Script for concatenating multiple HF datasets. Each dataset is expected to have
the same metadata (apart from the normalizer state) and be normalized. A new
normalizer is fitted on the concatenated dataset.

See `combine_processed_scene_datasets.py` for a similar script that also handles
different scene representations.
"""

import argparse
import logging
import os

from typing import Any

from datasets import Dataset, concatenate_datasets
from tqdm import tqdm

from steerable_scene_generation.utils.hf_dataset import (
    load_hf_dataset_with_metadata,
    normalize_all_scenes,
    save_hf_dataset_with_metadata,
    unnormalize_all_scenes,
)
from steerable_scene_generation.utils.min_max_scaler import (
    MinMaxScaler,
    fit_normalizer_hf,
)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input_files",
        nargs="+",
        type=str,
        required=True,
        help="List of input HF files to combine.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output file to save the combined dataset to.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=os.cpu_count(),
        help="The maximum number of parallel workers to use.",
    )
    args = parser.parse_args()
    input_files = args.input_files
    output_file = args.output_file
    max_workers = args.max_workers

    # Load the datasets.
    unnormalized_datasets: list[Dataset] = []
    metadatas: list[dict[str, Any]] = []
    for input_file in tqdm(input_files, desc="Loading datasets"):
        dataset, metadata = load_hf_dataset_with_metadata(input_file)

        # Load the normalizer.
        normalizer = MinMaxScaler(output_min=-1.0, output_max=1.0, clip=True)
        normalizer.load_serializable_state(metadata["normalizer_state"])
        if not normalizer.is_fitted:
            raise ValueError("Normalizer is not fitted!")

        # Unnormalize the dataset.
        combined_unnormalized_dataset = unnormalize_all_scenes(
            normalizer=normalizer, hf_dataset=dataset, num_procs=max_workers
        )
        combined_unnormalized_dataset.set_format("torch")

        unnormalized_datasets.append(combined_unnormalized_dataset)
        metadatas.append(metadata)

    if "subdataset_names" in metadatas[0]:
        assert len(metadatas[0]["subdataset_names"]) == len(
            input_files
        ), "The number of subdataset names must match the number of input files."
    if "subdataset_paths" in metadatas[0]:
        assert len(metadatas[0]["subdataset_paths"]) == len(
            input_files
        ), "The number of subdataset paths must match the number of input files."

    # Check that all metadatas are the same apart from the normalizer state.
    first_metadata = metadatas[0].copy()
    first_metadata.pop("normalizer_state", None)
    for i, metadata in enumerate(metadatas):
        # Ignore the normalizer state for comparison.
        metadata_copy = metadata.copy()
        metadata_copy.pop("normalizer_state", None)
        if metadata_copy != first_metadata:
            raise ValueError(f"Metadata object {i} is different from the first one.")

    # Concatenate the datasets.
    logging.info("Concatenating the datasets...")
    combined_unnormalized_dataset = concatenate_datasets(unnormalized_datasets)
    logging.info(f"Concatenated dataset size: {len(combined_unnormalized_dataset)}")

    # Fit a new normalizer on the modified dataset.
    logging.info("Fitting a new normalizer on the concatenated dataset...")
    combined_unnormalized_dataset.set_format("torch", columns=["scenes"])
    new_normalizer, normalizer_state = fit_normalizer_hf(
        hf_dataset=combined_unnormalized_dataset, num_proc=max_workers
    )
    logging.info("Fitted a new normalizer on the concatenated dataset.")

    # Normalize the modified dataset.
    normalized_filtered_dataset = normalize_all_scenes(
        normalizer=new_normalizer,
        hf_dataset=combined_unnormalized_dataset,
        num_procs=max_workers,
        batch_size=1,
    )
    logging.info("Normalized the concatenated dataset.")

    # Create new metadata.
    new_metadata = metadatas[0].copy()
    new_metadata["normalizer_state"] = normalizer_state

    if "subdataset_ranges" in new_metadata:
        # Update the subdataset ranges.
        new_subdataset_ranges = []
        current_idx = 0
        for dataset in unnormalized_datasets:
            new_subdataset_ranges.append([current_idx, current_idx + len(dataset)])
            current_idx += len(dataset)
        new_metadata["subdataset_ranges"] = new_subdataset_ranges

        logging.info("Updated the subdataset ranges.")

    # Save the concatenated dataset.
    logging.info(f"Saving concatenated dataset to {output_file}")
    save_hf_dataset_with_metadata(
        hf_dataset=normalized_filtered_dataset,
        metadata=new_metadata,
        dataset_path=output_file,
        num_procs=max_workers,
    )
    logging.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
