"""
Script for modifying the rotation representation of a HF dataset.
"""

import argparse
import copy
import logging
import os

from functools import partial

import torch

from steerable_scene_generation.algorithms.common.dataclasses import (
    RotationParametrization,
    SceneVecDescription,
)
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


def modify_scene_rotation_representation(
    item: dict,
    scene_vec_desc: SceneVecDescription,
    new_scene_vec_desc: SceneVecDescription,
) -> dict:
    scene: torch.Tensor = item["scenes"]  # Shape (N, V)

    # Convert the rotation to quaternion.
    quaternion = scene_vec_desc.get_quaternion(scene)

    # Convert the quaternion to the new rotation representation.
    new_rotation_vec = new_scene_vec_desc.quaternion_to_rotation_vec(quaternion)

    # Update the scene.
    translation_vec = scene_vec_desc.get_translation_vec(scene)
    model_path_vec = scene_vec_desc.get_model_path_vec(scene)
    new_scene = new_scene_vec_desc.get_scene_or_obj_from_components(
        translation_vec=translation_vec,
        rotation_vec=new_rotation_vec,
        model_path_vec=model_path_vec,
    )
    item["scenes"] = new_scene

    return item


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the HF dataset to modify.",
    )
    parser.add_argument(
        "output_dataset_path",
        type=str,
        help="Path to save the modified dataset to.",
    )
    parser.add_argument(
        "--target_rotation_representation",
        type=RotationParametrization,
        choices=list(RotationParametrization),
        required=True,
        help="Target rotation representation.",
    )
    parser.add_argument(
        "--not_normalize_one_hot_features",
        action="store_true",
        help="If true, will not normalize the one-hot features. Note that this needs "
        "to be enabled, even if all input datasets had their one-hot vector "
        "unnormalized.",
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
    target_rotation_representation = args.target_rotation_representation
    not_normalize_one_hot_features = args.not_normalize_one_hot_features
    max_workers = args.max_workers

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
    logging.info("Unnormalized the dataset.")

    # Get the scene vector description.
    scene_vec_desc = get_scene_vec_description_from_metadata(metadata=metadata)

    # Ensure that the new rotation representation is different from the current one.
    if target_rotation_representation == scene_vec_desc.rotation_parametrization:
        raise ValueError(
            "Target rotation representation is the same as the current one!"
        )

    # Create the new scene vector description.
    new_scene_vec_desc = SceneVecDescription(
        drake_package_map=scene_vec_desc.drake_package_map,
        static_directive=scene_vec_desc.static_directive,
        translation_vec_len=scene_vec_desc.translation_vec_len,
        rotation_parametrization=target_rotation_representation,
        model_paths=scene_vec_desc.model_paths,
        model_path_vec_len=scene_vec_desc.model_path_vec_len,
        welded_object_model_paths=scene_vec_desc.welded_object_model_paths,
    )

    # Modify the dataset.
    unnormalized_dataset.set_format("torch", columns=["scenes"])
    modified_unnormalized_dataset = unnormalized_dataset.map(
        partial(
            modify_scene_rotation_representation,
            scene_vec_desc=scene_vec_desc,
            new_scene_vec_desc=new_scene_vec_desc,
        ),
        num_proc=max_workers,
    )
    logging.info("Modified the dataset.")

    # Fit a new normalizer on the modified dataset.
    new_normalizer, new_normalizer_state = fit_normalizer_hf(
        hf_dataset=modified_unnormalized_dataset, num_proc=max_workers
    )
    logging.info("Fitted a new normalizer on the modified dataset.")

    if not_normalize_one_hot_features:
        logging.info("Not normalizing one-hot features...")

        # Exclude the one-hot model path vector from the normalizer. Assumes that
        # the model path vector comes last.
        model_path_vec_len = metadata["model_path_vec_len"]
        new_normalizer.params["scale"][-model_path_vec_len:] = 1.0
        new_normalizer.params["min"][-model_path_vec_len:] = 0.0

        # Get the updated normalizer state.
        new_normalizer_state = new_normalizer.get_serializable_state()

    # Normalize the modified dataset.
    normalized_modified_dataset = normalize_all_scenes(
        normalizer=new_normalizer,
        hf_dataset=modified_unnormalized_dataset,
        num_procs=max_workers,
        batch_size=1,
    )
    logging.info("Normalized the modified dataset.")

    # Update metadata.
    new_metadata = copy.deepcopy(metadata)
    new_metadata["normalizer_state"] = new_normalizer_state
    new_metadata["rotation_parametrization"] = target_rotation_representation.value
    new_metadata["is_one_hot_vector_normalized"] = not not_normalize_one_hot_features

    # Save the modified dataset.
    logging.info(f"Saving modified dataset to {output_dataset_path}")
    save_hf_dataset_with_metadata(
        hf_dataset=normalized_modified_dataset,
        metadata=new_metadata,
        dataset_path=output_dataset_path,
        num_procs=max_workers,
    )
    logging.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
