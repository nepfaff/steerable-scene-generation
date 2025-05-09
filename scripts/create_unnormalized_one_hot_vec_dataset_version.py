"""
Script for creating a version of a dataset whose one-hot model path vector is
unnormalized. The new dataset will be associated with a normalizer that has no impact
on the one-hot vector.
This is useful for approaches such as mixed diffusion whose discrete diffusion relies
on the discrete part being represented as one-hot vectors.
"""

import argparse
import os

from steerable_scene_generation.utils.hf_dataset import (
    load_hf_dataset_with_metadata,
    normalize_all_scenes,
    save_hf_dataset_with_metadata,
    unnormalize_all_scenes,
)
from steerable_scene_generation.utils.min_max_scaler import MinMaxScaler


def create_new_dataset(
    dataset_path: str, output_path: str, num_procs: int = os.cpu_count()
):
    """
    Create a new dataset whose one-hot model path vector is unnormalized.
    Processes the data in chunks to reduce memory usage.

    Args:
        dataset_path (str): Path to the input dataset.
        output_path (str): Path to the output dataset.
        num_procs (int): The number of processes to.
    """
    dataset, metadata = load_hf_dataset_with_metadata(dataset_path=dataset_path)

    # Create the normalizer.
    normalizer = MinMaxScaler(output_min=-1.0, output_max=1.0, clip=True)
    normalizer.load_serializable_state(metadata["normalizer_state"])

    # Unnormalize the dataset.
    unnormalized_dataset = unnormalize_all_scenes(
        normalizer=normalizer,
        hf_dataset=dataset,
        num_procs=num_procs,
    )

    # Exclude the one-hot model path vector from the normalizer. Assumes that the model
    # path vector comes last.
    model_path_vec_len = metadata["model_path_vec_len"]
    normalizer.params["scale"][-model_path_vec_len:] = 1.0
    normalizer.params["min"][-model_path_vec_len:] = 0.0

    # Normalize the dataset with the updated normalizer params.
    new_dataset = normalize_all_scenes(
        normalizer=normalizer,
        hf_dataset=unnormalized_dataset,
        num_procs=num_procs,
    )

    # Update the normalizer in the metadata.
    metadata["normalizer_state"] = normalizer.get_serializable_state()

    # Mark and save the new dataset.
    metadata["is_one_hot_vector_normalized"] = False
    save_hf_dataset_with_metadata(
        hf_dataset=new_dataset,
        metadata=metadata,
        dataset_path=output_path,
        num_procs=num_procs,
    )
    print(f"Dataset and metadata saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the input HF dataset that is fully normalized.",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to the new HF dataset whose one-hot model path vector is not "
        "normalized.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=os.cpu_count(),
        help="The maximum number of processes to use.",
    )

    args = parser.parse_args()

    # Create the new dataset.
    create_new_dataset(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        num_procs=args.max_workers,
    )


if __name__ == "__main__":
    main()
