import os
import re

import numpy as np
import torch

from pydrake.all import (
    AddDirectives,
    AddModel,
    AddWeld,
    ModelDirective,
    ModelDirectives,
    Rotation,
    Transform,
)
from pytorch3d.transforms import matrix_to_euler_angles

from steerable_scene_generation.algorithms.common.dataclasses import SceneVecDescription


def extract_base_link_name_from_sdf(sdf_path: str) -> str:
    """
    Extract the base link name from an SDF file.

    Args:
        sdf_path (str): Path to the SDF file.

    Returns:
        str: The name of the base link.

    Raises:
        ValueError: If the SDF file contains more than one link or no links.
    """
    with open(sdf_path, "r") as f:
        content = f.read()

    # Find all link elements and their names.
    link_matches = re.findall(r'<link\s+name=["\']([^"\']+)["\']', content)

    if not link_matches:
        raise ValueError(f"No links found in SDF file: {sdf_path}")

    if len(link_matches) > 1:
        raise ValueError(
            f"Multiple links found in SDF file: {sdf_path}. This is not supported."
        )

    return link_matches[0]


def extract_base_link_name_from_sdf_package_file(
    package_file_path: str, scene_vec_desc: SceneVecDescription
) -> str:
    """
    Extract the base link name from an SDF file.

    Args:
        package_file_path (str): Path to the package file.
        scene_vec_desc (SceneVecDescription): The scene vector description.

    Returns:
        str: The name of the base link.
    """
    if not package_file_path.startswith("package://") or not package_file_path.endswith(
        ".sdf"
    ):
        raise ValueError(f"Invalid package file path: {package_file_path}")

    # Split path.
    package_name = package_file_path.split("package://")[1].split("/")[0]
    path_in_package = "/".join(package_file_path.split("package://")[1].split("/")[1:])

    # Get the path to the package.
    package_path = scene_vec_desc.drake_package_map.GetPath(package_name)
    sdf_path = os.path.join(package_path, path_in_package)

    return extract_base_link_name_from_sdf(sdf_path)


def get_scene_directives(
    scenes: torch.Tensor | np.ndarray, scene_vec_desc: SceneVecDescription
) -> list[ModelDirectives]:
    """
    Creates and returns Drake directives for a batch of scenes:
    https://drake.mit.edu/doxygen_cxx/structdrake_1_1multibody_1_1parsing_1_1_model_directives.html

    These directives allow the scenes to be directly loaded into Drake for simulation.

    Args:
        scenes (torch.Tensor | np.ndarray): A batch of unormalized scenes. Shape
            (B, N, V).
        scene_vec_desc (SceneVecDescription): The scene vector description.

    Returns:
        list[ModelDirectives]: A list of scene directives of shape (B,).
    """
    if not len(scenes.shape) == 3:
        raise ValueError(f"Scenes must be a 3D tensor. Got shape {scenes.shape}.")

    if isinstance(scenes, torch.Tensor):
        scenes = scenes.cpu().detach().numpy()

    scene_directives = []
    for scene in scenes:
        directives = []

        # Add the static directive if it exists.
        if scene_vec_desc.static_directive is not None:
            directive = ModelDirective(
                add_directives=AddDirectives(file=scene_vec_desc.static_directive)
            )
            directives.append(directive)

        # Add all objects.
        object_idx = 0
        for i, obj in enumerate(scene):
            model_path = scene_vec_desc.get_model_path(
                i if scene_vec_desc.model_path_vec_len is None else obj
            )
            if model_path is None:
                # Skip empty objects.
                continue

            object_idx += 1

            translation = scene_vec_desc.get_translation_vec(obj).tolist()
            rotation_matrix = scene_vec_desc.get_rotation_matrix(torch.tensor(obj))
            euler_angles = matrix_to_euler_angles(rotation_matrix, convention="XYZ")
            euler_angles_degrees = np.rad2deg(euler_angles).tolist()

            if model_path.endswith(".sdf"):
                if model_path.startswith("package://"):
                    base_link_name = extract_base_link_name_from_sdf_package_file(
                        package_file_path=model_path, scene_vec_desc=scene_vec_desc
                    )
                else:
                    base_link_name = extract_base_link_name_from_sdf(model_path)
            else:
                raise NotImplementedError("Only SDF models are supported at this time.")

            # Create the model directive.
            if scene_vec_desc.is_welded_object(model_path):
                add_model_directive = ModelDirective(
                    add_model=AddModel(
                        name=f"object_{object_idx}",
                        file=model_path,
                    )
                )
                directives.append(add_model_directive)
                add_weld_directive = ModelDirective(
                    add_weld=AddWeld(
                        parent="world",
                        child=f"object_{object_idx}::{base_link_name}",
                        X_PC=Transform(
                            rotation=Rotation.Rpy(deg=euler_angles_degrees),
                            translation=translation,
                        ),
                    )
                )
                directives.append(add_weld_directive)
            else:
                add_model_directive = ModelDirective(
                    add_model=AddModel(
                        name=f"object_{object_idx}",
                        file=model_path,
                        default_free_body_pose={
                            base_link_name: Transform(
                                rotation=Rotation.Rpy(deg=euler_angles_degrees),
                                translation=translation,
                            )
                        },
                    )
                )
                directives.append(add_model_directive)

        model_directives = ModelDirectives(directives=directives)
        scene_directives.append(model_directives)

    return scene_directives
