import logging

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union

import numpy as np
import roma
import torch

from pydrake.all import (
    BodyIndex,
    DiagramBuilder,
    ModelInstanceIndex,
    MultibodyPlant,
    PackageMap,
    RigidTransform,
    SceneGraph,
)
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    matrix_to_quaternion,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
)
from typing_extensions import Self


class RotationParametrization(Enum):
    AXIS_ANGLE = "axis_angle"
    PROCRUSTES = "procrustes"
    QUATERNION = "quaternion"

    def __len__(self):
        if self == RotationParametrization.AXIS_ANGLE:
            return 3
        elif self == RotationParametrization.PROCRUSTES:
            return 9
        elif self == RotationParametrization.QUATERNION:
            return 4
        else:
            raise ValueError(f"Invalid rotation parametrization {self}.")

    @staticmethod
    def from_str(s: str):
        if s == "axis_angle":
            return RotationParametrization.AXIS_ANGLE
        elif s == "procrustes":
            return RotationParametrization.PROCRUSTES
        elif s == "quaternion":
            return RotationParametrization.QUATERNION
        else:
            raise ValueError(f"Invalid rotation parametrization {s}.")


@dataclass
class SceneVecDifference:
    """
    A dataclass representing the difference between two scene vectors.
    """

    translation: torch.Tensor
    """ (B, N, 4) """
    rotation: torch.Tensor
    """ (B, N, K) where K=3 if we compute the difference on SO(3) as the logarithm
    map. Otherwise it is the dimension of the rotation vector."""
    model_path: torch.Tensor | None

    def __post_init__(
        self,
    ):
        assert self.translation.ndim == 3, f"translation.ndim={self.translation.ndim}"
        assert self.rotation.ndim == 3, f"rotation.ndim={self.rotation.ndim}"
        assert (
            self.translation.shape[:2] == self.rotation.shape[:2]
        ), f"translation.shape={self.translation.shape}, rotation.shape={self.rotation.shape}"

    def scale(self, scalers: torch.Tensor) -> Self:
        """
        scalers is of shape (B,)
        """
        assert scalers.ndim == 1
        assert scalers.shape[0] == self.translation.shape[0]
        scalers_expand = scalers.unsqueeze(-1).unsqueeze(-1)
        return type(self)(
            translation=self.translation * scalers_expand,
            rotation=self.rotation * scalers_expand,
            model_path=(
                None if self.model_path is None else self.model_path * scalers_expand
            ),
        )

    def to_tensor(self) -> torch.Tensor:
        tensors = [self.translation, self.rotation]
        if self.model_path is not None:
            tensors.append(self.model_path)
        return torch.concat(tensors, dim=-1)


@dataclass(frozen=True)
class SceneVecDescription:
    """
    A dataclass to describe the the scene vector that is used for diffusion.

    The scene vector has the following structure: [translation, rotation, model_path],
    where the model_path is optional.
    """

    drake_package_map: PackageMap
    """The package map for resolving the model paths."""
    static_directive: Union[str, None]
    """A directive to specify additional static objects in the scene. These objects are
    not optimized over but are added whenever a plant is constructed. Note that this
    directive should not add positions to the plant (all objects must be welded)."""
    translation_vec_len: int
    """The length of the translation vector."""
    rotation_parametrization: RotationParametrization
    """The rotation parametrization that is used."""
    model_paths: List[str]
    """The model paths of the objects in the scene."""
    model_path_vec_len: Union[int, None]
    """The length of the model path vector. If None, the model paths are assumed to be
    ordered and can contain duplicates. In that case, the nth object in the scene is
    assumed to correspond to the nth model path in the model_paths list."""
    welded_object_model_paths: List[str] = field(default_factory=list)
    """The model paths of the welded objects in the scene. The models specified here
    must also be present in `model_paths`. This specifies whether a model is floating
    or welded."""

    def get_object_vec_len(self) -> int:
        model_path_vec_len = (
            0 if self.model_path_vec_len is None else self.model_path_vec_len
        )
        return (
            self.translation_vec_len
            + len(self.rotation_parametrization)
            + model_path_vec_len
        )

    def get_translation_vec_len(self) -> int:
        return self.translation_vec_len

    def get_rotation_vec_len(self) -> int:
        return len(self.rotation_parametrization)

    def get_model_path_vec_len(self) -> int:
        return self.model_path_vec_len

    def get_diff_vec_len(self) -> int:
        """
        The length of the difference between two object feature vectors.
        """
        rotation_diff_vec_len = len(self.rotation_parametrization)
        model_path_vec_len = (
            0 if self.model_path_vec_len is None else self.model_path_vec_len
        )
        return self.translation_vec_len + rotation_diff_vec_len + model_path_vec_len

    def get_translation_vec(self, scene_or_obj: torch.Tensor) -> torch.Tensor:
        return scene_or_obj[..., : self.translation_vec_len]

    def update_translation_vec(
        self, scene_or_obj: torch.Tensor, translation_vec: torch.Tensor
    ) -> None:
        assert translation_vec.shape[-1] == self.translation_vec_len
        scene_or_obj[..., : self.translation_vec_len] = translation_vec

    def get_rotation_vec(self, scene_or_obj: torch.Tensor) -> torch.Tensor:
        return scene_or_obj[
            ...,
            self.translation_vec_len : self.translation_vec_len
            + len(self.rotation_parametrization),
        ]

    def update_rotation_vec(
        self, scene_or_obj: torch.Tensor, rotation_vec: torch.Tensor
    ) -> None:
        assert scene_or_obj.shape[-1] == self.get_object_vec_len()
        assert rotation_vec.shape[-1] == len(self.rotation_parametrization)
        scene_or_obj[
            ...,
            self.translation_vec_len : self.translation_vec_len
            + len(self.rotation_parametrization),
        ] = rotation_vec

    def get_model_path_vec(self, scene_or_obj: torch.Tensor) -> Optional[torch.Tensor]:
        if self.model_path_vec_len is None:
            return None
        return scene_or_obj[
            ...,
            self.translation_vec_len
            + len(self.rotation_parametrization) : self.translation_vec_len
            + len(self.rotation_parametrization)
            + self.model_path_vec_len,
        ]

    def get_model_path_from_model_path_vec(
        self, model_path_vec: Union[torch.Tensor, np.ndarray]
    ) -> Union[str, None]:
        """
        Args:
            model_path_vec (Union[torch.Tensor, np.ndarray]): A one-hot vector
                representing  a model path. Shape (M+1,) where M is the number of unique
                model paths. Empty objects are represented by the (M+1)th class.

        Returns:
            Union[str, None]: The model path corresponding to the one-hot vector. None
                if the one-hot vector is empty.
        """
        num_models = len(self.model_paths)
        if not len(model_path_vec) == num_models + 1:
            raise ValueError(
                f"Length of model_path_vec must be equal to num_models + 1. "
                f"Expected length: {num_models + 1}, but got: {len(model_path_vec)}. "
                "This error indicates that the provided one-hot vector does not match "
                "the expected format for model paths, which includes an additional "
                "entry for empty objects."
            )

        # Convert numpy array to torch tensor if necessary.
        if isinstance(model_path_vec, np.ndarray):
            model_path_vec = torch.tensor(model_path_vec)

        model_path_idx = int(torch.argmax(model_path_vec).item())
        if model_path_idx == num_models:
            # Empty object.
            return None
        return self.model_paths[model_path_idx]

    def get_model_path_vec_from_model_path(
        self, model_path: str | None
    ) -> torch.Tensor:
        """
        Args:
            model_path (str | None): The model path to get the one-hot vector for.
                None represents an empty object.

        Returns:
            torch.Tensor: The one-hot vector for the model path of shape (M+1,) where M
                is the number of unique model paths. Empty objects are represented by
                the (M+1)th class.
        """
        if model_path is None:
            # For empty object, return one-hot vector with 1 in the last position.
            return torch.cat((torch.zeros(len(self.model_paths)), torch.tensor([1])))

        model_path_vec = torch.tensor([model_path == mp for mp in self.model_paths])
        # Append a zero to the end to represent the empty object.
        return torch.cat((model_path_vec, torch.tensor([0])), dim=0)

    def get_model_path(self, obj_or_idx: torch.Tensor | np.ndarray | int) -> str | None:
        """
        Args:
            obj_or_idx (torch.Tensor | int): If model_path_vec_len is None, this must be
                an index. If model_path_vec_len is not None, this must be a tensor.

        Returns:
            Union[str, None]: The model path of the object. None corresponds to the
                empty object.
        """
        # Convert to tensor if is numpy array.
        if isinstance(obj_or_idx, np.ndarray):
            obj_or_idx = torch.tensor(obj_or_idx)

        if (self.model_path_vec_len is None and not isinstance(obj_or_idx, int)) or (
            self.model_path_vec_len is not None
            and not isinstance(obj_or_idx, torch.Tensor)
        ):
            raise ValueError(
                "obj_or_idx must be an index if model_path_vec_len is None and a "
                "tensor if model_path_vec_len is not None."
            )

        if self.model_path_vec_len is None:
            # Model paths are ordered and can contain duplicates.
            return self.model_paths[obj_or_idx]

        if not obj_or_idx.dim() == 1:
            raise ValueError("obj must have shape (V,)")

        model_path_vec = self.get_model_path_vec(obj_or_idx)
        return self.get_model_path_from_model_path_vec(model_path_vec)

    def replace_masked_objects_with_empty(
        self, scene: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Replace the objects in the scene that are masked with empty objects. Note that
        this only modifies the model path vector. The continuous part stays the same.

        Args:
            scene (torch.Tensor): The scene(s) to replace the objects in.
                Shape (..., N, V).
            mask (torch.Tensor): The boolean mask to replace the objects in.
                Shape (..., N).

        Returns:
            torch.Tensor: The scene with the objects replaced with empty objects.
                Shape (..., N, V).
        """
        # Validate the mask.
        assert mask.shape == scene.shape[:-1]
        assert mask.dtype == torch.bool

        # Replace the objects in the scene that are masked with empty objects.
        model_path_vec = self.get_model_path_vec(scene)
        model_path_vec[mask] = self.get_model_path_vec_from_model_path(None).to(
            scene.device
        )
        return self.get_scene_or_obj_from_components(
            translation_vec=self.get_translation_vec(scene),
            rotation_vec=self.get_rotation_vec(scene),
            model_path_vec=model_path_vec,
        )

    def is_welded_object(self, model_path: str | None) -> bool:
        """
        Args:
            model_path (str | None): The model path to check.

        Returns:
            bool: True if the model path is a welded object, False otherwise.
        """
        if model_path is None:
            return False
        return model_path in self.welded_object_model_paths

    def get_scene_without_model_path(self, scene_or_obj: torch.Tensor) -> torch.Tensor:
        return scene_or_obj[
            ..., : self.translation_vec_len + len(self.rotation_parametrization)
        ]

    def get_rotation_matrix(self, scene_or_obj: torch.Tensor) -> torch.Tensor:
        rotation_vec = self.get_rotation_vec(scene_or_obj)
        if self.rotation_parametrization == RotationParametrization.PROCRUSTES:
            rotation_vec_matrix = rotation_vec.reshape(*rotation_vec.shape[:-1], 3, 3)
            try:
                rotation_matrix = roma.special_procrustes(rotation_vec_matrix)
            except:
                num_nans = torch.isnan(rotation_vec_matrix).sum()
                num_infs = torch.isinf(rotation_vec_matrix).sum()
                logging.error(
                    f"rotation_vec_matrix contains {num_nans} nans and "
                    f"{num_infs} infs!"
                )
                logging.warning(
                    "Attempting to fix the issue by replacing the nans and infs with "
                    "valid floats."
                )
                rotation_vec_matrix = torch.nan_to_num(
                    rotation_vec_matrix, nan=0.0, posinf=1e10, neginf=-1e10
                )
                rotation_matrix = roma.special_procrustes(rotation_vec_matrix)

        elif self.rotation_parametrization == RotationParametrization.AXIS_ANGLE:
            rotation_matrix = axis_angle_to_matrix(rotation_vec)
        elif self.rotation_parametrization == RotationParametrization.QUATERNION:
            rotation_matrix = quaternion_to_matrix(rotation_vec)
        else:
            raise ValueError(
                f"Invalid rotation parametrization {self.rotation_parametrization}."
            )
        return rotation_matrix

    def get_quaternion(self, scene_or_obj: torch.Tensor) -> torch.Tensor:
        rotation_vec = self.get_rotation_vec(scene_or_obj)
        if self.rotation_parametrization == RotationParametrization.PROCRUSTES:
            return matrix_to_quaternion(self.get_rotation_matrix(scene_or_obj))
        elif self.rotation_parametrization == RotationParametrization.AXIS_ANGLE:
            return axis_angle_to_quaternion(rotation_vec)
        elif self.rotation_parametrization == RotationParametrization.QUATERNION:
            return rotation_vec
        else:
            raise ValueError(
                f"Invalid rotation parametrization {self.rotation_parametrization}."
            )

    def quaternion_to_rotation_vec(self, quaternion: torch.Tensor) -> torch.Tensor:
        """
        Convert a quaternion to a rotation vector.

        Args:
            quaternion (torch.Tensor): The quaternion to convert. Shape (..., 4).

        Returns:
            torch.Tensor: The rotation vector. Shape (..., R) where R is the dimension
                of the rotation vector.
        """
        if not quaternion.shape[-1] == 4:
            raise ValueError(
                f"Quaternion must have shape (..., 4). Got {quaternion.shape}."
            )

        if self.rotation_parametrization == RotationParametrization.PROCRUSTES:
            rotation_vec = quaternion_to_matrix(quaternion).flatten(start_dim=-2)
        elif self.rotation_parametrization == RotationParametrization.AXIS_ANGLE:
            rotation_vec = quaternion_to_axis_angle(quaternion)
        elif self.rotation_parametrization == RotationParametrization.QUATERNION:
            rotation_vec = quaternion
        else:
            raise ValueError(
                f"Invalid rotation parametrization {self.rotation_parametrization}."
            )
        return rotation_vec

    def get_scene_or_obj_from_components(
        self,
        translation_vec: torch.Tensor,
        rotation_vec: torch.Tensor,
        model_path_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (self.model_path_vec_len is None) != (model_path_vec is None):
            raise ValueError(
                f"Mismatch between model_path_vec_len ({self.model_path_vec_len}) and "
                f"model_path_vec presence ({model_path_vec is not None}). "
                "Ensure that model_path_vec is provided when model_path_vec_len is not None."
            )

        components = [translation_vec, rotation_vec]

        if model_path_vec is not None:
            components.append(model_path_vec)

        scene = torch.cat(components, dim=-1)
        if not scene.shape[-1] == self.get_object_vec_len():
            raise ValueError(
                f"Scene has {scene.shape[-1]} elements, but expected "
                f"{self.get_object_vec_len()}."
            )
        return scene

    def calc_difference(
        self, scene_or_obj_from: torch.Tensor, scene_or_obj_to: torch.Tensor
    ) -> SceneVecDifference:
        """
        Calculate the difference between two scene description vectors.

        In the most naive case, the difference is scene_or_obj_to - scene_or_obj_from.
        """
        assert scene_or_obj_from.shape == scene_or_obj_to.shape

        translation_diff = self.get_translation_vec(
            scene_or_obj_to
        ) - self.get_translation_vec(scene_or_obj_from)
        rotation_diff = self.get_rotation_vec(scene_or_obj_to) - self.get_rotation_vec(
            scene_or_obj_from
        )
        model_path_to = self.get_model_path_vec(scene_or_obj_to)
        model_path_from = self.get_model_path_vec(scene_or_obj_from)
        if model_path_to is None or model_path_from is None:
            model_path_diff = None
        else:
            model_path_diff = model_path_to - model_path_from
        return SceneVecDifference(
            translation=translation_diff,
            rotation=rotation_diff,
            model_path=model_path_diff,
        )

    def calc_sum(
        self, scene_or_obj: torch.Tensor, diff: SceneVecDifference
    ) -> torch.Tensor:
        """
        The inverse of calc_diff. calc_sum(scene_or_obj_from, diff) returns
        scene_or_diff_to, such that calc_diff(scene_or_obj_from, scene_or_obj_to) = diff.
        """
        translation_to = self.get_translation_vec(scene_or_obj) + diff.translation
        rotation_vec_to = self.get_rotation_vec(scene_or_obj) + diff.rotation
        model_path_from = self.get_model_path_vec(scene_or_obj)
        if model_path_from is None or diff.model_path is None:
            model_path_to = None
        else:
            model_path_to = model_path_from + diff.model_path
        return self.get_scene_or_obj_from_components(
            translation_vec=translation_to,
            rotation_vec=rotation_vec_to,
            model_path_vec=model_path_to,
        )

    def to_diff_from_tensor(self, diff_tensor: torch.Tensor) -> SceneVecDifference:
        """
        `diff_tensor` contains the concatenation of translation, rotation, model_path,
        etc. This function splits `diff_tensor` and store the result in the returned
        SceneVecDifference object.
        """
        assert diff_tensor.shape[-1] == self.get_diff_vec_len()
        vec_len = 0
        translation = diff_tensor[..., : self.translation_vec_len]
        vec_len += self.translation_vec_len
        rotation = diff_tensor[
            ..., vec_len : vec_len + len(self.rotation_parametrization)
        ]
        vec_len += len(self.rotation_parametrization)
        model_path = (
            None
            if self.model_path_vec_len is None or self.model_path_vec_len == 0
            else diff_tensor[..., vec_len : vec_len + self.model_path_vec_len]
        )
        vec_len += self.model_path_vec_len if self.model_path_vec_len is not None else 0
        return SceneVecDifference(
            translation=translation, rotation=rotation, model_path=model_path
        )

    def __getstate__(self):
        state = self.__dict__.copy()

        # Replace non-pickable PackageMap object with pickable data.
        package_names = self.drake_package_map.GetPackageNames()
        if "drake_models" in package_names:
            package_names.remove("drake_models")
        package_paths = [
            self.drake_package_map.GetPath(package_name)
            for package_name in package_names
        ]
        state["drake_package_map"] = dict(zip(package_names, package_paths))

        return state

    def __setstate__(self, state):
        # Restore the state.
        self.__dict__.update(state)

        # Reconstruct the non-pickable PackageMap object.
        drake_package_map = PackageMap()
        for package_name, package_path in state["drake_package_map"].items():
            try:
                drake_package_map.Add(package_name, package_path)
            except:
                print(f"Failed to add package {package_name} with path {package_path}.")
        object.__setattr__(self, "drake_package_map", drake_package_map)


@dataclass
class PlantSceneGraphCache:
    """
    A cache object for the plant and scene graph. This is used to avoid recreating
    the plant and scene graph when the objects in the scene have not changed.
    """

    diagram: DiagramBuilder
    plant: MultibodyPlant
    scene_graph: SceneGraph
    rigid_body_indices: List[BodyIndex]
    """The rigid body indices of the objects in the scene."""
    object_model_paths: List[str]
    """The model paths of the objects in the scene."""
    model_indices: List[ModelInstanceIndex]
    """The model indices of the objects in the scene."""


@dataclass
class PlantSceneGraphResult:
    """
    A dataclass containing the results of creating a plant and scene graph from a scene.
    """

    plant: MultibodyPlant
    """The MultibodyPlant created from the scene."""
    scene_graph: SceneGraph
    """The SceneGraph created from the scene."""
    rigid_body_indices: List[BodyIndex | None]
    """The rigid body indices of the objects in the scene. None for empty objects."""
    object_model_paths: List[str | None]
    """The model paths of the objects in the scene. None for empty objects."""
    model_indices: List[ModelInstanceIndex | None]
    """The model indices of the objects in the scene. None for empty objects."""
    object_transforms: List[RigidTransform | None]
    """The transforms of the objects in the scene. None for empty objects."""
