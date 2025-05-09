"""
Script for filtering a HF dataset based on a set of criteria.
Currently, only filtering based on object translation ranges is supported.

This can be very useful for mixed precision training where large translation ranges
might make it hard to distinguish between close values.
"""

import argparse
import copy
import logging
import os

from functools import partial

import torch

from steerable_scene_generation.algorithms.common.dataclasses import SceneVecDescription
from steerable_scene_generation.utils.hf_dataset import (
    get_scene_vec_description_from_metadata,
    load_hf_dataset_with_metadata,
    normalize_all_scenes,
    save_hf_dataset_with_metadata,
    unnormalize_all_scenes,
)
from steerable_scene_generation.utils.min_max_scaler import (
    MinMaxScaler,
    fit_normalizer_hf,
)


def filter_scene_by_translation_range(
    scene_data: dict,
    scene_vec_desc: SceneVecDescription,
    x_min: float = float("-inf"),
    x_max: float = float("inf"),
    y_min: float = float("-inf"),
    y_max: float = float("inf"),
    z_min: float = float("-inf"),
    z_max: float = float("inf"),
) -> bool:
    """
    Filters a scene based on whether all objects in the scene have translations
    within the specified ranges.

    Args:
        scene_data (dict): The scene data containing the scenes.
        scene_vec_desc: The scene vector description.
        x_min (float): Minimum x coordinate allowed.
        x_max (float): Maximum x coordinate allowed.
        y_min (float): Minimum y coordinate allowed.
        y_max (float): Maximum y coordinate allowed.
        z_min (float): Minimum z coordinate allowed.
        z_max (float): Maximum z coordinate allowed.

    Returns:
        bool: True if all objects in the scene are within the specified ranges, False
        otherwise.
    """
    scene = scene_data["scenes"]
    if not len(scene.shape) == 2:
        raise ValueError("Scene must be a 2D tensor.")

    # Extract translations for all objects in the scene.
    translations = scene_vec_desc.get_translation_vec(scene)

    # Check if all objects are within the specified ranges.
    x_in_range = (translations[:, 0] >= x_min) & (translations[:, 0] <= x_max)
    y_in_range = (translations[:, 1] >= y_min) & (translations[:, 1] <= y_max)
    z_in_range = (translations[:, 2] >= z_min) & (translations[:, 2] <= z_max)

    # Return True only if all objects are within all specified ranges.
    return torch.all(x_in_range & y_in_range & z_in_range).item()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the HF dataset to filter.",
    )
    parser.add_argument(
        "output_dataset_path",
        type=str,
        help="Path to save the filtered dataset to.",
    )
    parser.add_argument(
        "--x_min",
        type=float,
        default=float("-inf"),
        help="Minimum x coordinate allowed for objects in meters.",
    )
    parser.add_argument(
        "--x_max",
        type=float,
        default=float("inf"),
        help="Maximum x coordinate allowed for objects in meters.",
    )
    parser.add_argument(
        "--y_min",
        type=float,
        default=float("-inf"),
        help="Minimum y coordinate allowed for objects in meters.",
    )
    parser.add_argument(
        "--y_max",
        type=float,
        default=float("inf"),
        help="Maximum y coordinate allowed for objects in meters.",
    )
    parser.add_argument(
        "--z_min",
        type=float,
        default=float("-inf"),
        help="Minimum z coordinate allowed for objects in meters.",
    )
    parser.add_argument(
        "--z_max",
        type=float,
        default=float("inf"),
        help="Maximum z coordinate allowed for objects in meters.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=os.cpu_count(),
        help="The maximum number of parallel workers to use.",
    )
    args = parser.parse_args()
    dataset_path = args.dataset_path
    output_dataset_path = args.output_dataset_path
    max_workers = args.max_workers
    x_min = args.x_min
    x_max = args.x_max
    y_min = args.y_min
    y_max = args.y_max
    z_min = args.z_min
    z_max = args.z_max

    # Validate inputs.
    if x_min > x_max:
        raise ValueError("x_min must be less than x_max.")
    if y_min > y_max:
        raise ValueError("y_min must be less than y_max.")
    if z_min > z_max:
        raise ValueError("z_min must be less than z_max.")

    # Load dataset.
    logging.info(f"Loading dataset from {dataset_path}")
    base_dataset, metadata = load_hf_dataset_with_metadata(dataset_path)
    normalizer = MinMaxScaler(output_min=-1.0, output_max=1.0, clip=True)
    normalizer.load_serializable_state(metadata["normalizer_state"])
    if not normalizer.is_fitted:
        raise ValueError("Normalizer is not fitted!")
    logging.info("Loaded the dataset and normalizer.")

    # Prepare unnormalized dataset.
    unnormalized_dataset = unnormalize_all_scenes(
        normalizer=normalizer, hf_dataset=base_dataset, num_procs=max_workers
    )
    unnormalized_dataset.set_format("torch")
    logging.info("Unnormalized the dataset.")

    # Get the scene vector description.
    scene_vec_desc = get_scene_vec_description_from_metadata(metadata=metadata)

    # Filter the dataset.
    logging.info("Filtering dataset based on translation ranges:")
    logging.info(f" X range: [{args.x_min}, {args.x_max}]")
    logging.info(f" Y range: [{args.y_min}, {args.y_max}]")
    logging.info(f" Z range: [{args.z_min}, {args.z_max}]")
    filtered_unnormalized_dataset = unnormalized_dataset.filter(
        partial(
            filter_scene_by_translation_range,
            scene_vec_desc=scene_vec_desc,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
        ),
        num_proc=max_workers,
    )
    logging.info("Filtered the dataset based on translation ranges.")

    # Fit a new normalizer on the modified dataset.
    new_normalizer, normalizer_state = fit_normalizer_hf(
        hf_dataset=filtered_unnormalized_dataset, num_proc=max_workers
    )
    logging.info("Fitted a new normalizer on the filtered dataset.")

    # Normalize the modified dataset.
    normalized_filtered_dataset = normalize_all_scenes(
        normalizer=new_normalizer,
        hf_dataset=filtered_unnormalized_dataset,
        num_procs=max_workers,
        batch_size=1,
    )
    logging.info("Normalized the filtered dataset.")

    # Update metadata to include the new normalizer and filtering information.
    new_metadata = copy.deepcopy(metadata)
    new_metadata["normalizer_state"] = normalizer_state
    new_metadata["filtering"] = {
        "x_range": [x_min, x_max],
        "y_range": [y_min, y_max],
        "z_range": [z_min, z_max],
    }

    # Save the filtered dataset.
    logging.info(f"Original dataset size: {len(base_dataset)}")
    logging.info(f"Filtered dataset size: {len(filtered_unnormalized_dataset)}")
    logging.info(f"Saving filtered dataset to {output_dataset_path}")
    save_hf_dataset_with_metadata(
        hf_dataset=normalized_filtered_dataset,
        metadata=new_metadata,
        dataset_path=output_dataset_path,
        num_procs=max_workers,
    )
    logging.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
