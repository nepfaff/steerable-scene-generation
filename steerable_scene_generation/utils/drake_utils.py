import logging
import os

from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Context,
    DiagramBuilder,
    DiscreteContactApproximation,
    ModelInstanceIndex,
    MultibodyPlant,
    PackageMap,
    Parser,
    RigidTransform,
)

from steerable_scene_generation.algorithms.common.dataclasses import (
    PlantSceneGraphCache,
    PlantSceneGraphResult,
    SceneVecDescription,
)
from steerable_scene_generation.utils.caching import have_objects_in_scene_changed

console_logger = logging.getLogger(__name__)


def make_package_map(package_name: str, package_file_path: str) -> PackageMap:
    package_map = PackageMap()
    package_file_abs_path = os.path.abspath(os.path.expanduser(package_file_path))
    package_map.Add(package_name, os.path.dirname(package_file_abs_path))
    return package_map


def create_plant_and_scene_graph_from_scene(
    scene: np.ndarray,
    builder: DiagramBuilder,
    scene_vec_desc: SceneVecDescription,
    weld_objects: bool,
    time_step: float = 0.0,
) -> PlantSceneGraphResult:
    """
    Create a MultibodyPlant and SceneGraph from a scene.

    Args:
        scene (np.ndarray): The scene to create the plant from. Shape (N, T+R+M) where
            N is the number of objects, T is the translation vector length, R is the
            rotation vector length, and M is the model path vector length. This is the
            unnormalized scene.
        builder (DiagramBuilder): The diagram builder to add the plant and scene graph
            to.
        scene_vec_desc (SceneVecDescription): The scene vector description.
        weld_objects (bool): Whether to weld the objects to the world frame.
        time_step (float): The time step of the plant.

    Returns:
        PlantSceneGraphResult: A dataclass containing the plant, scene graph, rigid body
            indices, object model paths, model indices, and object transforms.
    """
    # Convert BFloat16 tensors to float32 to avoid type errors.
    if isinstance(scene, torch.Tensor) and scene.dtype == torch.bfloat16:
        scene = scene.to(torch.float32)

    # Setup plant.
    plant: MultibodyPlant
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    if time_step > 0.0:
        plant.set_discrete_contact_approximation(DiscreteContactApproximation.kLagged)
    parser = Parser(plant)
    parser.package_map().AddMap(scene_vec_desc.drake_package_map)
    parser.SetAutoRenaming(True)

    # Add static models.
    if scene_vec_desc.static_directive is not None:
        if scene_vec_desc.static_directive.startswith("package://"):
            parser.AddModelsFromUrl(scene_vec_desc.static_directive)
        else:
            parser.AddModels(scene_vec_desc.static_directive)

    # Add scene models.
    rigid_body_indices = []
    object_model_paths = []
    model_indices = []
    object_transforms = []
    for i, obj in enumerate(scene):
        translation = scene_vec_desc.get_translation_vec(obj)
        rotation_matrix = (
            scene_vec_desc.get_rotation_matrix(torch.tensor(obj))
            .to(torch.float32)
            .numpy()
        )

        model_path = scene_vec_desc.get_model_path(
            i if scene_vec_desc.model_path_vec_len is None else obj
        )

        # Create transform.
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = translation
        transform = RigidTransform(transform)

        if model_path is None:
            # Skip empty objects.
            rigid_body_indices.append(None)
            object_model_paths.append(None)
            model_indices.append(None)
            object_transforms.append(None)
            continue

        if model_path.startswith("package://"):
            models: List[ModelInstanceIndex] = parser.AddModelsFromUrl(model_path)
        else:
            models: List[ModelInstanceIndex] = parser.AddModels(model_path)
        assert len(models) == 1
        model = models[0]

        body_indices = plant.GetBodyIndices(model)
        assert len(body_indices) == 1
        body_index = body_indices[0]
        body = plant.get_body(body_index)
        rigid_body_indices.append(body.index())
        object_model_paths.append(model_path)
        model_indices.append(model)
        object_transforms.append(transform)

        # Set scene model transforms.
        for body_index in body_indices:
            body = plant.get_body(body_index)
            if weld_objects or scene_vec_desc.is_welded_object(model_path):
                plant.WeldFrames(plant.world_frame(), body.body_frame(), transform)
            else:
                plant.SetDefaultFreeBodyPose(body, transform)

    plant.Finalize()

    return PlantSceneGraphResult(
        plant=plant,
        scene_graph=scene_graph,
        rigid_body_indices=rigid_body_indices,
        object_model_paths=object_model_paths,
        model_indices=model_indices,
        object_transforms=object_transforms,
    )


def update_scene_poses_from_plant(
    scene: torch.Tensor,
    plant: MultibodyPlant,
    plant_context: Context,
    model_indices: List[ModelInstanceIndex],
    scene_vec_desc: SceneVecDescription,
) -> torch.Tensor:
    """
    Updates the object poses of a scene tensor from the plant context.

    Args:
        scene (torch.Tensor): The unormalized scene tensor to update. Shape (N, D) where
            N is the number of objects and D is the dimension of each object.
        plant (MultibodyPlant): The plant.
        plant_context (Context): The plant context.
        model_indices (List[ModelInstanceIndex]): The model indices of the objects in
            the scene. The order must correspond to the order of the objects in the
            scene tensor.
        scene_vec_desc (SceneVecDescription): The scene vector description.

    Returns:
        torch.Tensor: The updated unormalized scene tensor.
    """
    assert len(scene) == len(model_indices)

    updated_scene = scene.clone()
    for obj, model_idx in zip(updated_scene, model_indices):
        if model_idx is None:
            # No need to update empty objects.
            continue

        # Update the pose with the one stored in the context.
        translation = torch.tensor(plant.GetPositions(plant_context, model_idx)[4:7])
        quaternion = torch.tensor(plant.GetPositions(plant_context, model_idx)[:4])
        # Only floating bodies have a translation and rotation.
        if translation.numel() == 0:
            # Keep the original translation.
            translation = scene_vec_desc.get_translation_vec(obj)
        if quaternion.numel() == 0:
            # Keep the original rotation.
            rotation = scene_vec_desc.get_rotation_vec(obj)
        else:
            rotation = scene_vec_desc.quaternion_to_rotation_vec(quaternion)

        model_path_vec = (
            scene_vec_desc.get_model_path_vec(obj)
            if scene_vec_desc.model_path_vec_len is not None
            else None
        )

        obj[:] = scene_vec_desc.get_scene_or_obj_from_components(
            translation_vec=translation,
            rotation_vec=rotation,
            model_path_vec=model_path_vec,
        ).to(obj.device)

    return updated_scene


def create_plant_and_scene_graph_from_scene_with_cache(
    scene: Union[torch.Tensor, np.ndarray],
    scene_vec_desc: SceneVecDescription,
    time_step: float = 0.0,
    cache: Optional[PlantSceneGraphCache] = None,
) -> Tuple[PlantSceneGraphCache, Context, Context]:
    """
    Creates a new plant and scene graph from the scene tensor `scene`. If `cache` is
    not None and the objects in the scene have not changed, the plant and scene graph
    from the cache are re-used.

    Args:
        scene: A tensor of shape (N, D) where N is the number of objects in the scene
            and D is the dimension of each object. This is the unnormalized scene.
        scene_vec_desc (SceneVecDescription): The scene vector description.
        time_step: The time step of the plant.
        cache: The PlantSceneGraphCache. If None or if the objects in the scene have
            changed, the plant and scene graph are recreated.

    Returns:
        A tuple (cache, context, plant_context) where:
            - cache: A PlantSceneGraphCache object.
            - context: The context of the diagram.
            - plant_context: The context of the plant.
    """
    if cache is not None and not have_objects_in_scene_changed(
        scene=scene, cache=cache, scene_vec_desc=scene_vec_desc
    ):
        # Re-use the cached plant and scene graph.
        diagram = cache.diagram
        plant = cache.plant
        scene_graph = cache.scene_graph
        rigid_body_indices = cache.rigid_body_indices

        # Update the body poses.
        context = diagram.CreateDefaultContext()
        for i, obj in enumerate(scene):
            # Skip empty objects
            if rigid_body_indices[i] is None:
                continue

            obj_np = (
                obj.cpu().detach().numpy() if isinstance(obj, torch.Tensor) else obj
            )
            translation = scene_vec_desc.get_translation_vec(obj_np)
            rotation_matrix = scene_vec_desc.get_rotation_matrix(
                torch.tensor(obj_np)
            ).numpy()

            transform = np.eye(4)
            transform[:3, :3] = rotation_matrix
            transform[:3, 3] = translation

            plant_context = plant.GetMyContextFromRoot(context)
            model_path = (
                scene_vec_desc.get_model_path(obj_np)
                if scene_vec_desc.model_path_vec_len is not None
                else i
            )
            if not scene_vec_desc.is_welded_object(model_path):
                plant.SetFreeBodyPose(
                    context=plant_context,
                    body=plant.get_body(rigid_body_indices[i]),
                    X_PB=RigidTransform(transform),
                )
            else:
                console_logger.warning(
                    f"Can't update pose of welded object {model_path}. The multibody "
                    "plant might differ from the scene vector due to using caching!"
                )

    else:
        # Create new diagram.
        builder = DiagramBuilder()

        result = create_plant_and_scene_graph_from_scene(
            scene=(
                scene.cpu().detach().numpy()
                if isinstance(scene, torch.Tensor)
                else scene
            ),
            builder=builder,
            scene_vec_desc=scene_vec_desc,
            weld_objects=False,
            time_step=time_step,
        )
        diagram = builder.Build()

        context = diagram.CreateDefaultContext()
        plant_context = result.plant.GetMyContextFromRoot(context)

        # Create a new cache.
        cache = PlantSceneGraphCache(
            diagram=diagram,
            plant=result.plant,
            scene_graph=result.scene_graph,
            rigid_body_indices=result.rigid_body_indices,
            object_model_paths=result.object_model_paths,
            model_indices=result.model_indices,
        )

    return cache, context, plant_context
