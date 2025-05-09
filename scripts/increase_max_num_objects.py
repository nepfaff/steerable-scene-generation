"""
Script for increasing the maximum number of objects of a dataset by padding with empty
object tokens.
"""

import argparse
import os

import torch

from steerable_scene_generation.utils.hf_dataset import (
    load_hf_dataset_with_metadata,
    normalize_all_scenes,
    save_hf_dataset_with_metadata,
    unnormalize_all_scenes,
)
from steerable_scene_generation.utils.min_max_scaler import MinMaxScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the dataset to increase the max number of objects for.",
    )
    parser.add_argument(
        "output_dataset_path",
        type=str,
        help="Path to the new dataset.",
    )
    parser.add_argument(
        "--new_max_num",
        type=int,
        required=True,
        help="The new maximum number of objects per scene.",
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
    new_max_obj_num = args.new_max_num
    max_workers = args.max_workers

    dataset, metadata = load_hf_dataset_with_metadata(dataset_path)
    current_max_num_objects_per_scene = metadata["max_num_objects_per_scene"]
    assert (
        new_max_obj_num > current_max_num_objects_per_scene
    ), f"Require {new_max_obj_num} > {current_max_num_objects_per_scene}"

    # Load the normalizer.
    normalizer = MinMaxScaler(output_min=-1.0, output_max=1.0, clip=True)
    normalizer.load_serializable_state(metadata["normalizer_state"])
    if not normalizer.is_fitted:
        raise ValueError("Normalizer is not fitted!")

    # Unormalize the scenes.
    unnormalized_dataset = unnormalize_all_scenes(
        normalizer=normalizer, hf_dataset=dataset, num_procs=max_workers
    )
    unnormalized_dataset.set_format("torch")

    # Create the empty objects to add to each scene. The [empty] object is represented
    # as the last category.
    additional_empty_objects = new_max_obj_num - current_max_num_objects_per_scene
    empty_vec = torch.zeros_like(unnormalized_dataset[0]["scenes"][0])  # Shape (V,)
    empty_vec[-1] = 1.0
    additional_object_vecs = torch.stack(
        [empty_vec] * additional_empty_objects
    )  # Shape (E, V)

    def add_empty_object_vecs(item):
        scene = item["scenes"]  # Shape (O, V)
        new_scene = torch.concat([scene, additional_object_vecs])  # Shape (O+E, V)
        item["scenes"] = new_scene
        return item

    # Add the empty object vectors to each scene.
    extended_unnormalized_dataset = unnormalized_dataset.map(
        add_empty_object_vecs,
        num_proc=max_workers,
        batched=False,
        desc="Adding empty object vectors to scenes",
    )

    # Normalize the scenes.
    extended_dataset = normalize_all_scenes(
        normalizer=normalizer,
        hf_dataset=extended_unnormalized_dataset,
        num_procs=max_workers,
    )

    # Update the metadata.
    metadata["max_num_objects_per_scene"] = new_max_obj_num

    # Save the new dataset.
    save_hf_dataset_with_metadata(
        hf_dataset=extended_dataset,
        metadata=metadata,
        dataset_path=output_dataset_path,
        num_procs=max_workers,
    )


if __name__ == "__main__":
    main()
