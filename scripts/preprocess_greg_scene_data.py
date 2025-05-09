"""
Script for pre-processing the Greg scene data as produced by our procedural generation
pipeline at https://github.com/nepfaff/spatial_scene_grammars.
"""

import argparse
import os
import pickle
import xml.etree.ElementTree as ET

from functools import partial
from typing import List, Union

import numpy as np
import roma
import torch

from pytorch3d.transforms import matrix_to_axis_angle
from tqdm.contrib.concurrent import process_map


def extract_obj_path_from_sdf(sdf_content: str) -> Union[str, None]:
    root = ET.fromstring(sdf_content)
    obj_path = root.find(".//mesh/uri")
    if obj_path is not None:
        return obj_path.text
    return None


def get_model_path_from_obj_path(obj_path: str, model_parent_dir_path: str) -> str:
    return (
        obj_path
        if obj_path.startswith("package://")
        else os.path.join(model_parent_dir_path, obj_path)
    )


def process_scene(
    scene: list[dict],
    model_paths: list[str],
    model_parent_dir_path: str,
    rotation_parametrization: str,
    fixed: bool,
    max_num_objects_per_scene: int,
    min_num_objects: int,
) -> tuple[Union[np.ndarray, None], int]:
    """
    Process a single scene.

    Args:
        scene: The scene to process.
        model_paths: List of model paths.
        model_parent_dir_path: Path to the parent directory containing model files.
        rotation_parametrization: Type of rotation parametrization to use.
        fixed: Whether objects in scenes are fixed.
        max_num_objects_per_scene: Maximum number of objects per scene. Used for
            empty object padding.
        min_num_objects: Minimum number of objects per scene. Used for filtering.

    Returns:
        A tuple containing the processed scene vector and stats (objects removed).
    """
    object_vecs = []
    num_objects_removed = 0

    for obj in scene:
        transform = obj["transform"]
        if np.any(np.isnan(transform)) or np.any(np.isinf(transform)):
            num_objects_removed += 1
            continue

        translation = transform[:3, 3]
        if rotation_parametrization == "axis_angle":
            rotation = matrix_to_axis_angle(torch.tensor(transform[:3, :3])).numpy()
        elif rotation_parametrization == "procrustes":
            rotation = (
                roma.special_procrustes(torch.tensor(transform[:3, :3]))
                .numpy()
                .flatten()
            )
        else:
            raise ValueError(
                f"Invalid rotation parametrization: {rotation_parametrization}"
            )
        model_path = get_model_path_from_obj_path(
            obj["model_path"], model_parent_dir_path
        )

        object_vec = [translation, rotation]
        if not fixed:
            # One hot model path vec. This includes the [empty] object at the last
            # slot.
            model_path_vec = [0] * (len(model_paths) + 1)  # Shape (O+1,)
            model_path_vec[model_paths.index(model_path)] = 1
            object_vec.append(model_path_vec)
        object_vec = np.concatenate(object_vec)

        object_vecs.append(object_vec)

    if len(object_vecs) < min_num_objects:
        return None, num_objects_removed

    # Concatenate object vecs and pad to num_objects_per_scene.
    if not fixed and len(object_vecs) < max_num_objects_per_scene:
        # Represent [empty] objects as the last category.
        empty_vec = np.zeros_like(object_vecs[0])
        empty_vec[-1] = 1.0
        object_vecs += [empty_vec] * (max_num_objects_per_scene - len(object_vecs))

    # Represent scene as stacked object vecs.
    scene_vec = np.stack(object_vecs, axis=0)  # Shape (N, O+1)
    return scene_vec, num_objects_removed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scenes_pickle_path",
        type=str,
        help="Path to the scene pickle file.",
    )
    parser.add_argument(
        "processed_pickle_path",
        type=str,
        help="Path to save the processed scene pickle file.",
    )
    parser.add_argument(
        "--model_parent_dir_path",
        type=str,
        default="data",
        help="Path to the parent directory containing the model files directory.",
    )
    parser.add_argument(
        "--fixed",
        action="store_true",
        help="Whether the objects in the scenes are fixed. For use with "
        + "SceneDiffuserUnetFixedObjects",
    )
    parser.add_argument(
        "--rotation_parametrization",
        type=str,
        default="procrustes",
        choices=["axis_angle", "procrustes"],
        help="Rotation parametrization to use.",
    )
    parser.add_argument(
        "--min_num_objects",
        type=int,
        default=3,
        help="Minimum number of objects per scene.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count(),
        help="Number of worker processes for parallel processing.",
    )
    args = parser.parse_args()
    scenes_pickle_path = args.scenes_pickle_path
    processed_pickle_path = args.processed_pickle_path
    model_parent_dir_path = args.model_parent_dir_path
    fixed = args.fixed
    rotation_parametrization = args.rotation_parametrization
    min_num_objects = args.min_num_objects
    num_workers = args.num_workers

    with open(scenes_pickle_path, "rb") as f:
        raw_scenes: List[dict] = pickle.load(f)

    # Determine number of unique models.
    model_paths = []
    if fixed:
        for obj in raw_scenes[0]:
            model_path = get_model_path_from_obj_path(
                obj["model_path"], model_parent_dir_path
            )
            model_paths.append(model_path)
    else:
        for scene in raw_scenes:
            for obj in scene:
                model_path = get_model_path_from_obj_path(
                    obj["model_path"], model_parent_dir_path
                )
                if model_path not in model_paths:
                    model_paths.append(model_path)

    # Determine max and mean number of objects in a scene.
    lengths = [len(scene) for scene in raw_scenes]
    max_num_objects_per_scene = max(lengths)
    mean_num_objects_per_scene = np.mean(lengths)

    if fixed:
        # Ensure that all scenes have the same number of objects.
        fixed_len_scenes = []
        for scene in raw_scenes:
            if len(scene) != max_num_objects_per_scene:
                print(
                    f"Removed scene with {len(scene)} objects."
                    f"Expected {max_num_objects_per_scene}."
                )
            else:
                fixed_len_scenes.append(scene)
        raw_scenes = fixed_len_scenes

    if fixed:
        # Sort based on model path. The nth object in a scene corresponds to the nth
        # model path.
        model_paths.sort()
        for scene in raw_scenes:
            scene.sort(
                key=lambda obj: model_paths.index(
                    os.path.join(model_parent_dir_path, obj["model_path"])
                )
            )

    # Re-format data using parallel processing.
    process_scene_partial = partial(
        process_scene,
        model_paths=model_paths,
        model_parent_dir_path=model_parent_dir_path,
        rotation_parametrization=rotation_parametrization,
        fixed=fixed,
        max_num_objects_per_scene=max_num_objects_per_scene,
        min_num_objects=min_num_objects,
    )
    results = process_map(
        process_scene_partial,
        raw_scenes,
        max_workers=num_workers,
        chunksize=1000,
    )

    # Process results.
    scenes = []
    num_objects_removed = 0
    num_scenes_removed = 0
    for scene_vec, objects_removed in results:
        if scene_vec is None:
            num_scenes_removed += 1
        else:
            scenes.append(scene_vec)
        num_objects_removed += objects_removed

    print("Final number of scenes:", len(scenes))
    print(f"Removed {num_objects_removed} objects and {num_scenes_removed} scenes.")

    # Save processed scenes.
    save_dict = {
        "rotation_parametrization": rotation_parametrization,
        "translation_vec_len": 3,
        "model_path_vec_len": len(model_paths) + 1 if not fixed else None,
        "max_num_objects_per_scene": max_num_objects_per_scene,
        "mean_num_objects_per_scene": mean_num_objects_per_scene,
    }
    print(save_dict)
    save_dict["scenes"] = scenes
    save_dict["model_paths"] = model_paths
    os.makedirs(os.path.dirname(processed_pickle_path), exist_ok=True)
    with open(processed_pickle_path, "wb") as f:
        pickle.dump(save_dict, f)


if __name__ == "__main__":
    main()
