"""
Script for adding a static/ welded object to a dataset. It does the following:
- Adds a new object category for the welded object
- Adds the welded object to each scene exactly once at the specified pose (same pose
  for all scenes)
"""

import argparse
import copy
import json
import os

from functools import partial

import torch

from datasets import Dataset

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


def add_static_object_to_scene(
    scene_data: dict,
    scene_vec_desc: SceneVecDescription,
    new_scene_vec_desc: SceneVecDescription,
    static_model_path: str,
    pose: torch.Tensor,
) -> dict:
    """
    Adds a static/welded object to the scene data.

    Args:
        scene_data (dict): The original scene data containing the scenes.
        scene_vec_desc (SceneVecDescription): The scene vector description for the
            original scene.
        new_scene_vec_desc (SceneVecDescription): The scene vector description for the
            new static object.
        static_model_path (str): The model path of the new static object.
        pose (torch.Tensor): A tensor containing the translation and rotation (quaternion)
            of the new object. The first three elements represent the translation
            (x, y, z), and the last four elements represent the quaternion (x, y, z, w)
            for rotation.

    Returns:
        dict: The updated scene data with the new static object added.
    """
    scene = scene_data["scenes"]
    if not len(scene.shape) == 2:
        raise ValueError("Scene must be a 2D tensor.")

    scene_without_model_path = scene_vec_desc.get_scene_without_model_path(scene)
    model_path_vecs = scene_vec_desc.get_model_path_vec(scene)

    # Extend the one-hot vector for each existing object.
    new_model_path_vecs = []
    for model_path_vec in model_path_vecs:
        model_path = scene_vec_desc.get_model_path_from_model_path_vec(model_path_vec)
        new_model_path_vec = new_scene_vec_desc.get_model_path_vec_from_model_path(
            model_path
        )
        new_model_path_vecs.append(new_model_path_vec)
    new_model_path_vecs = torch.stack(new_model_path_vecs, dim=0)
    extended_scene = torch.concatenate(
        [scene_without_model_path, new_model_path_vecs], dim=1
    )

    # Add the new object.
    static_model_path_vec = new_scene_vec_desc.get_model_path_vec_from_model_path(
        static_model_path
    )
    static_object = new_scene_vec_desc.get_scene_or_obj_from_components(
        translation_vec=pose[:3],
        rotation_vec=new_scene_vec_desc.quaternion_to_rotation_vec(pose[3:]),
        model_path_vec=static_model_path_vec,
    )
    extended_scene_with_static_object = torch.concatenate(
        [extended_scene, static_object.unsqueeze(0)], dim=0
    )

    scene_data["scenes"] = extended_scene_with_static_object
    return scene_data


def add_static_object_to_dataset(
    dataset: Dataset,
    scene_vec_desc: SceneVecDescription,
    model_path: str,
    pose: torch.Tensor,
    max_workers: int,
) -> tuple[Dataset, SceneVecDescription]:
    """
    Adds a static/ welded object to a dataset.

    Args:
        dataset (Dataset): The dataset to add the welded object to.
        scene_vec_desc (SceneVecDescription): The scene vector description for the
            input dataset.
        model_path (str): The model path to add. Can be a package path.
        pose (torch.Tensor): The pose to add the welded object to.
        max_workers (int): The maximum number of parallel workers to use.

    Returns:
        tuple[Dataset, SceneVecDescription]: The modified dataset with the welded
            object added and the new scene vector description.
    """
    # Create the new scene vector description.
    new_scene_vec_desc = SceneVecDescription(
        drake_package_map=scene_vec_desc.drake_package_map,
        static_directive=scene_vec_desc.static_directive,
        translation_vec_len=scene_vec_desc.translation_vec_len,
        rotation_parametrization=scene_vec_desc.rotation_parametrization,
        model_paths=scene_vec_desc.model_paths + [model_path],
        model_path_vec_len=scene_vec_desc.model_path_vec_len + 1,
    )

    # Add the welded object to the dataset.
    modified_dataset = dataset.map(
        partial(
            add_static_object_to_scene,
            scene_vec_desc=scene_vec_desc,
            new_scene_vec_desc=new_scene_vec_desc,
            static_model_path=model_path,
            pose=pose,
        ),
        num_proc=max_workers,
    )

    return modified_dataset, new_scene_vec_desc


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the HF dataset to add the welded object to.",
    )
    parser.add_argument(
        "output_dataset_path",
        type=str,
        help="Path to save the modified dataset to.",
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="The model path to add. Can be a package path.",
    )
    parser.add_argument(
        "--pose",
        type=json.loads,
        default=None,
        help="Pose to add the welded object to. If not specified, the object will be "
        "added to the origin. The pose has format [x, y, z, qw, qx, qy, qz].",
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
    model_path = args.model_path
    pose = torch.tensor(
        args.pose if args.pose is not None else [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    )
    max_workers = args.max_workers

    # Load dataset.
    base_dataset, metadata = load_hf_dataset_with_metadata(dataset_path)
    normalizer = MinMaxScaler(output_min=-1.0, output_max=1.0, clip=True)
    normalizer.load_serializable_state(metadata["normalizer_state"])
    if not normalizer.is_fitted:
        raise ValueError("Normalizer is not fitted!")

    # Prepare unnormalized dataset.
    unnormalized_dataset = unnormalize_all_scenes(
        normalizer=normalizer, hf_dataset=base_dataset, num_procs=max_workers
    )
    unnormalized_dataset.set_format("torch")

    # Get the scene vector description.
    scene_vec_desc = get_scene_vec_description_from_metadata(metadata=metadata)

    # Add the welded object to the dataset.
    modified_dataset, new_scene_vec_desc = add_static_object_to_dataset(
        dataset=unnormalized_dataset,
        scene_vec_desc=scene_vec_desc,
        model_path=model_path,
        pose=pose,
        max_workers=max_workers,
    )

    # Update the metadata.
    new_metadata = copy.deepcopy(metadata)
    new_metadata["model_paths"] = new_scene_vec_desc.model_paths
    new_metadata["model_path_vec_len"] = new_scene_vec_desc.model_path_vec_len
    new_metadata["welded_object_model_paths"] = list(
        set(metadata.get("welded_object_model_paths", []) + [model_path])
    )
    new_metadata["max_num_objects_per_scene"] = (
        metadata["max_num_objects_per_scene"] + 1
    )

    # Fit a new normalizer on the modified dataset.
    new_normalizer, normalizer_state = fit_normalizer_hf(
        hf_dataset=modified_dataset, num_proc=max_workers
    )
    new_metadata["normalizer_state"] = normalizer_state

    # Normalize the modified dataset.
    normalized_modified_dataset = normalize_all_scenes(
        normalizer=new_normalizer,
        hf_dataset=modified_dataset,
        num_procs=max_workers,
        batch_size=1,
    )

    # Save the modified dataset.
    save_hf_dataset_with_metadata(
        hf_dataset=normalized_modified_dataset,
        metadata=new_metadata,
        dataset_path=output_dataset_path,
        num_procs=max_workers,
    )


if __name__ == "__main__":
    main()
