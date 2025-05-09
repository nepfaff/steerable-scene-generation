import numpy as np
import torch

from pydrake.all import (
    BodyIndex,
    QueryObject,
    RigidTransform,
    SignedDistancePair,
    Simulator,
)

from steerable_scene_generation.algorithms.common.dataclasses import SceneVecDescription
from steerable_scene_generation.utils.drake_utils import (
    create_plant_and_scene_graph_from_scene_with_cache,
)


def generate_empty_object_inpainting_masks(
    scenes: torch.Tensor, scene_vec_desc: SceneVecDescription
) -> torch.Tensor:
    """
    Generate inpainting masks for empty objects in a batch of scenes.

    This function creates boolean masks indicating which objects in the
    provided scenes are empty.

    Args:
        scenes (torch.Tensor): A tensor representing a batch of scenes,
            with shape (B, N, V), where B is the batch size, N is the
            number of objects, and V is the object feature vector length. The scenes
            are assumed to be unnormalized.
        scene_vec_desc (SceneVecDescription): A description of the scene
            vector structure, used to determine the model paths of the objects.

    Returns:
        tuple[torch.Tensor, list[int]]: A tuple containing:
            - A boolean tensor of the same shape as `scenes`,
              indicating which objects are empty (True) and which are not (False).
            - The number of empty objects for each scene in the batch. Shape (B,).
    """
    masks = torch.zeros_like(scenes, dtype=torch.bool)  # Shape (B, N, V)
    batch_size, num_objects = scenes.shape[0], scenes.shape[1]
    empty_object_numbers = []

    for b in range(batch_size):
        empty_object_number = 0
        for n in range(num_objects):
            model_path = scene_vec_desc.get_model_path(scenes[b, n])
            if model_path is None:
                # Object is empty. Mask both continuous and discrete parts.
                masks[b, n, :] = True
                empty_object_number += 1
        empty_object_numbers.append(empty_object_number)

    return masks, empty_object_numbers


def generate_non_penetration_inpainting_masks(
    scenes: torch.Tensor,
    scene_vec_desc: SceneVecDescription,
    threshold: float = 0.0,
    exclude_welded_objects: bool = False,
) -> tuple[torch.Tensor, list[float]]:
    """
    Generate non-penetration inpainting masks for a batch of scenes.

    This function evaluates a batch of scenes to determine which objects are
    penetrating each other.

    Args:
        scenes (torch.Tensor): A tensor representing a batch of scenes,
            with shape (B, N, V), where B is the batch size, N is the
            number of objects, and V is the object feature vector length. The scenes
            are assumed to be unnormalized.
        scene_vec_desc (SceneVecDescription): A description of the scene
            vector structure.
        threshold (float, optional): The threshold for considering objects as
            penetrating each other in meters. Positive values indicate separation
            and negative values indicate penetration.
        exclude_welded_objects (bool, optional): Whether to exclude welded objects
            from the inpainting mask.
        return_total_penetration_distance (bool, optional): Whether to return the
            total penetration distance for each scene.

    Returns:
        tuple[torch.Tensor, list[float]]: A tuple containing:
            - A boolean tensor of the same shape as `scenes`,
              indicating which objects are penetrating (True) and which are not (False).
            - The total penetration distance for each scene in meters. Shape (B,).
    """
    masks = torch.zeros_like(scenes, dtype=torch.bool)  # Shape (B, N, V)
    scenes_np = scenes.detach().cpu().numpy()  # Shape (B, N, V)
    total_penetration_distances = []

    for i, scene in enumerate(scenes_np):
        # Create the diagram for the scene.
        cache, context, _ = create_plant_and_scene_graph_from_scene_with_cache(
            scene=scene, scene_vec_desc=scene_vec_desc
        )
        plant = cache.plant
        scene_graph = cache.scene_graph
        scene_graph_context = scene_graph.GetMyContextFromRoot(context)
        query_object: QueryObject = scene_graph.get_query_output_port().Eval(
            scene_graph_context
        )
        inspector = query_object.inspector()

        # Mapping from body index to object index in the scene vector.
        body_idx_to_object_idx: dict[BodyIndex, int] = {
            body_idx: i for i, body_idx in enumerate(cache.rigid_body_indices)
        }

        # Get all negative distances between the objects in the scene. These are the
        # penetration distances.
        signed_distance_pairs: list[
            SignedDistancePair
        ] = query_object.ComputeSignedDistancePairwiseClosestPoints(
            max_distance=threshold
        )

        # Mask the objects that are penetrating.
        for pair in signed_distance_pairs:
            frameA_id = inspector.GetFrameId(pair.id_A)
            frameB_id = inspector.GetFrameId(pair.id_B)

            bodyA = plant.GetBodyFromFrameId(frameA_id)
            bodyB = plant.GetBodyFromFrameId(frameB_id)

            objectA_idx = body_idx_to_object_idx[bodyA.index()]
            objectB_idx = body_idx_to_object_idx[bodyB.index()]

            if not exclude_welded_objects or bodyA.is_floating():
                masks[i, objectA_idx] = True
            if not exclude_welded_objects or bodyB.is_floating():
                masks[i, objectB_idx] = True

        # Negate to get distance in penetration.
        total_penetration_distance = -sum(
            [pair.distance for pair in signed_distance_pairs]
        )
        total_penetration_distances.append(total_penetration_distance)

    return masks, total_penetration_distances


def generate_static_equilibrium_inpainting_masks_with_sim(
    scenes: torch.Tensor,
    scene_vec_desc: SceneVecDescription,
    simulation_duration: float,
    time_step: float,
    translation_threshold: float = 0.0,
    rotation_threshold: float = 0.0,
) -> torch.Tensor:
    """
    Generate inpainting masks for static equilibrium by simulating the scene
    and identifying objects whose poses have changed beyond specified thresholds.

    Args:
        scenes (torch.Tensor): A tensor representing the scenes to be analyzed.
            Shape (B, N, V) where B is the batch size, N is the number of objects,
            and V is the vector size for each object. The scenes are assumed to be
            unnormalized.
        scene_vec_desc (SceneVecDescription): Description of the scene vector.
        simulation_duration (float): Duration for which the scene is simulated in
            seconds.
        time_step (float): Time step for the simulation in seconds.
        translation_threshold (float, optional): Threshold for translation changes
            to consider an object as having moved.
        rotation_threshold (float, optional): Threshold for rotation changes to
            consider an object as having moved.

    Returns:
        tuple[torch.Tensor, list[int]]: A tuple containing:
            - A boolean tensor of the same shape as `scenes`,
              indicating which objects have changed position beyond the specified
              thresholds.
            - The number of non-static equilibrium objects for each scene in the batch.
              Shape (B,).
    """
    masks = torch.zeros_like(scenes, dtype=torch.bool)  # Shape (B, N, V)
    scenes_np = scenes.detach().cpu().numpy()  # Shape (B, N, V)
    non_static_equilibrium_object_numbers = []

    for i, scene in enumerate(scenes_np):
        # Create the diagram for the scene.
        cache, context, _ = create_plant_and_scene_graph_from_scene_with_cache(
            scene=scene, scene_vec_desc=scene_vec_desc, time_step=time_step
        )
        plant = cache.plant
        plant_context = plant.GetMyContextFromRoot(context)

        # Mapping from body index to object index in the scene vector.
        body_idx_to_object_idx: dict[BodyIndex, int] = {
            body_idx: i for i, body_idx in enumerate(cache.rigid_body_indices)
        }

        # Save the initial object positions.
        body_idx_to_initial_pose: dict[BodyIndex, RigidTransform] = {}
        for body_idx in cache.rigid_body_indices:
            if body_idx is None:
                # Ignore empty objects.
                continue
            body = plant.get_body(body_idx)
            body_idx_to_initial_pose[body_idx] = body.EvalPoseInWorld(plant_context)

        # Simulate the scene.
        simulator = Simulator(cache.diagram, context)
        simulator.AdvanceTo(simulation_duration)

        # Mask the objects whose pose has changed by more than the threshold.
        non_static_equilibrium_object_number = 0
        for body_idx, initial_pose in body_idx_to_initial_pose.items():
            body = plant.get_body(body_idx)
            current_pose = body.EvalPoseInWorld(plant_context)
            if (
                np.linalg.norm(initial_pose.translation() - current_pose.translation())
                > translation_threshold
            ):
                masks[i, body_idx_to_object_idx[body_idx]] = True

            initial_quat = initial_pose.rotation().ToQuaternion()
            current_quat = current_pose.rotation().ToQuaternion()
            # Calculate angular distance between quaternions.
            dot_product = abs(
                initial_quat.w() * current_quat.w()
                + initial_quat.x() * current_quat.x()
                + initial_quat.y() * current_quat.y()
                + initial_quat.z() * current_quat.z()
            )
            # Clamp to valid domain for arccos.
            dot_product = np.clip(dot_product, -1.0, 1.0)
            angle_diff = 2 * np.arccos(dot_product)
            if angle_diff > rotation_threshold:
                masks[i, body_idx_to_object_idx[body_idx]] = True

            if masks[i, body_idx_to_object_idx[body_idx]].any():
                non_static_equilibrium_object_number += 1
        non_static_equilibrium_object_numbers.append(
            non_static_equilibrium_object_number
        )

    return masks, non_static_equilibrium_object_numbers


def generate_static_equilibrium_inpainting_masks_with_heuristic(
    scenes: torch.Tensor,
    scene_vec_desc: SceneVecDescription,
    distance_threshold: float = 0.0,
) -> tuple[torch.Tensor, list[int]]:
    """
    Generate inpainting masks for objects that are not in static equilibrium
    using a heuristic based on minimum distances between objects.

    This function identifies objects that are "floating" (not in contact with
    any other object) and thus cannot be in static equilibrium.

    Args:
        scenes (torch.Tensor): A tensor representing a batch of scenes,
            with shape (B, N, V), where B is the batch size, N is the
            number of objects, and V is the object feature vector length. The scenes
            are assumed to be unnormalized.
        scene_vec_desc (SceneVecDescription): A description of the scene
            vector structure.
        distance_threshold (float, optional): The threshold for considering objects
            as floating. Objects with minimum distance greater than this threshold
            are considered not in static equilibrium.

    Returns:
        tuple[torch.Tensor, list[int]]: A tuple containing:
            - A boolean tensor of the same shape as `scenes`,
              indicating which objects are not in static equilibrium (True).
            - The number of non-static equilibrium objects for each scene in the batch.
              Shape (B,).
    """
    masks = torch.zeros_like(scenes, dtype=torch.bool)  # Shape (B, N, V)
    scenes_np = scenes.detach().cpu().numpy()  # Shape (B, N, V)
    non_static_equilibrium_object_numbers = []

    for i, scene in enumerate(scenes_np):
        # Create the diagram for the scene.
        cache, context, _ = create_plant_and_scene_graph_from_scene_with_cache(
            scene=scene, scene_vec_desc=scene_vec_desc
        )
        plant = cache.plant
        scene_graph = cache.scene_graph
        scene_graph_context = scene_graph.GetMyContextFromRoot(context)
        query_object: QueryObject = scene_graph.get_query_output_port().Eval(
            scene_graph_context
        )
        inspector = query_object.inspector()

        # Mapping from body index to object index in the scene vector.
        body_idx_to_object_idx: dict[BodyIndex, int] = {
            body_idx: i for i, body_idx in enumerate(cache.rigid_body_indices)
        }

        signed_distance_pairs: list[
            SignedDistancePair
        ] = query_object.ComputeSignedDistancePairwiseClosestPoints(max_distance=5.0)

        # Track the minimum distance for each body.
        body_idx_to_min_distance: dict[BodyIndex, float] = {}

        # Calculate minimum distances for each body.
        for pair in signed_distance_pairs:
            frameA_id = inspector.GetFrameId(pair.id_A)
            frameB_id = inspector.GetFrameId(pair.id_B)

            bodyA = plant.GetBodyFromFrameId(frameA_id)
            bodyB = plant.GetBodyFromFrameId(frameB_id)

            # Count penetrations as zero distance.
            body_idx_to_min_distance[bodyA.index()] = max(
                0.0,
                min(
                    body_idx_to_min_distance.get(bodyA.index(), float("inf")),
                    pair.distance,
                ),
            )
            body_idx_to_min_distance[bodyB.index()] = max(
                0.0,
                min(
                    body_idx_to_min_distance.get(bodyB.index(), float("inf")),
                    pair.distance,
                ),
            )

        # Mark objects that are floating.
        non_static_equilibrium_count = 0
        for body_idx, min_distance in body_idx_to_min_distance.items():
            if min_distance > distance_threshold and body_idx in body_idx_to_object_idx:
                obj_idx = body_idx_to_object_idx[body_idx]
                if not masks[i, obj_idx].any():  # Only count each object once
                    masks[i, obj_idx] = True
                    non_static_equilibrium_count += 1

        non_static_equilibrium_object_numbers.append(non_static_equilibrium_count)

    return masks, non_static_equilibrium_object_numbers


def generate_physical_feasibility_inpainting_masks(
    scenes: torch.Tensor,
    scene_vec_desc: SceneVecDescription,
    non_penetration_threshold: float,
    use_sim: bool = False,
    sim_duration: float | None = None,
    sim_time_step: float | None = None,
    sim_translation_threshold: float | None = None,
    sim_rotation_threshold: float | None = None,
    static_equilibrium_distance_threshold: float | None = None,
    exclude_welded_objects_from_non_penetration_masking: bool = False,
) -> tuple[torch.Tensor, list[float], list[int]]:
    """
    Generate inpainting masks for physical feasibility in a batch of scenes.

    This function evaluates a batch of scenes to determine which objects
    are physically feasible based on non-penetration and static equilibrium
    criteria.
    The objects that are in penetration are removed before checking for static
    equilibrium with the remaining objects.

    Args:
        scenes (torch.Tensor): A tensor representing a batch of scenes,
            with shape (B, N, V), where B is the batch size, N is the
            number of objects, and V is the object feature vector length. The scenes
            are assumed to be unnormalized.
        scene_vec_desc (SceneVecDescription): A description of the scene
            vector structure, used to determine the model paths of the objects.
        non_penetration_threshold (float): The threshold for considering
            objects as penetrating each other in meters.
        use_sim (bool): Whether to use simulation for static equilibrium checks.
            Otherwise, a heuristic based on minimum distances between objects is used.
        sim_duration (float | None): Duration of the simulation in seconds if
            use_sim is True.
        sim_time_step (float | None): Time step for the simulation in seconds if
            use_sim is True.
        sim_translation_threshold (float | None): Threshold for translation movement
            in the simulation in meters if use_sim is True.
        sim_rotation_threshold (float | None): Threshold for rotation movement in
            the simulation if use_sim is True.
        static_equilibrium_distance_threshold (float | None): Distance threshold
            for static equilibrium checks if use_sim is False.
        exclude_welded_objects_from_non_penetration_masking (bool): Whether to exclude
            welded objects from the non-penetration masking.

    Returns:
        tuple[torch.Tensor, list[float], list[int]]: A tuple containing:
            - A boolean tensor indicating which objects are not physically feasible of
              shape (B, N, V).
            - The penetration distances for each scene. Shape (B,).
            - The number of non-static equilibrium objects for each scene. Shape (B,).
    """
    # Clone the scenes to avoid modifying the original scenes.
    scenes = scenes.clone()

    # Validate the inputs.
    if use_sim and (
        sim_duration is None
        or sim_time_step is None
        or sim_translation_threshold is None
        or sim_rotation_threshold is None
    ):
        raise ValueError(
            "If use_sim is True, sim_duration, sim_time_step, "
            "sim_translation_threshold, and sim_rotation_threshold must be provided."
        )
    elif not use_sim and static_equilibrium_distance_threshold is None:
        raise ValueError(
            "static_equilibrium_distance_threshold must be provided if use_sim is False."
        )

    # Generate the non-penetration masks.
    (
        non_penetration_masks,
        penetration_distances,
    ) = generate_non_penetration_inpainting_masks(
        scenes=scenes,
        scene_vec_desc=scene_vec_desc,
        threshold=non_penetration_threshold,
        exclude_welded_objects=exclude_welded_objects_from_non_penetration_masking,
    )

    # Replace masked objects with empty ones for the static equilibrium check.
    object_level_masks = non_penetration_masks.any(dim=2)  # Shape (B, N)
    scenes_with_empty_objects = scene_vec_desc.replace_masked_objects_with_empty(
        scene=scenes, mask=object_level_masks
    )  # Shape (B, N, V)

    # Generate the static equilibrium masks.
    if use_sim:
        (
            static_equilibrium_masks,
            non_static_equilibrium_object_numbers,
        ) = generate_static_equilibrium_inpainting_masks_with_sim(
            scenes=scenes_with_empty_objects,
            scene_vec_desc=scene_vec_desc,
            simulation_duration=sim_duration,
            time_step=sim_time_step,
            translation_threshold=sim_translation_threshold,
            rotation_threshold=sim_rotation_threshold,
        )
    else:
        (
            static_equilibrium_masks,
            non_static_equilibrium_object_numbers,
        ) = generate_static_equilibrium_inpainting_masks_with_heuristic(
            scenes=scenes_with_empty_objects,
            scene_vec_desc=scene_vec_desc,
            distance_threshold=static_equilibrium_distance_threshold,
        )

    # Combine the masks.
    inpainting_masks = torch.logical_or(non_penetration_masks, static_equilibrium_masks)

    return (
        inpainting_masks,
        penetration_distances,
        non_static_equilibrium_object_numbers,
    )
