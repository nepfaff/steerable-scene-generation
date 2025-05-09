import multiprocessing

from functools import partial
from typing import List

import numpy as np
import torch

from pydrake.all import (
    DiagramBuilder,
    ModelInstanceIndex,
    QueryObject,
    SignedDistancePair,
)

from steerable_scene_generation.algorithms.common.dataclasses import SceneVecDescription
from steerable_scene_generation.utils.drake_utils import (
    create_plant_and_scene_graph_from_scene,
)


def compute_total_scene_penetration(
    scene: np.ndarray, scene_vec_desc: SceneVecDescription
) -> float:
    """
    Compute the total scene penetration as the sum of the penetration distances of all
    objects in the scene. A higher number means more penetration.

    Args:
        scene (np.ndarray): The unormalized scene tensor. Shape (N, V) where N is the
            number of objects and V is the object feature vector length.
        scene_vec_desc (SceneVecDescription): The scene vector description.

    Returns:
        float: The total scene penetraton.
    """
    # Obtain the plant and scene graph.
    builder = DiagramBuilder()
    result = create_plant_and_scene_graph_from_scene(
        scene=scene,
        builder=builder,
        scene_vec_desc=scene_vec_desc,
        weld_objects=False,
    )
    plant = result.plant
    scene_graph = result.scene_graph
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    scene_graph_context = scene_graph.GetMyContextFromRoot(context)
    query_object: QueryObject = scene_graph.get_query_output_port().Eval(
        scene_graph_context
    )
    inspector = query_object.inspector()

    # Get all signed distance pairs of objects in penetration (negative distance).
    try:
        signed_distance_pairs: List[
            SignedDistancePair
        ] = query_object.ComputeSignedDistancePairwiseClosestPoints(max_distance=-1e-12)
    except:
        # See https://github.com/RobotLocomotion/drake/issues/21673 for why this might
        # occur.
        print("Computing signed distance failed, returning zero scene penetration.")
        return 0.0

    # Sum up the penetration distances for all collision pairs involving a floating
    # body.
    scene_penetration = 0.0
    for pair in signed_distance_pairs:
        frameA_id = inspector.GetFrameId(pair.id_A)
        frameB_id = inspector.GetFrameId(pair.id_B)

        modelA = ModelInstanceIndex(inspector.GetFrameGroup(frameA_id))
        modelB = ModelInstanceIndex(inspector.GetFrameGroup(frameB_id))
        if plant.num_positions(modelA) == 0 and plant.num_positions(modelB) == 0:
            # Two welded objects in penetration. Continue.
            continue

        # Subtract as pair.distance is negative.
        scene_penetration -= pair.distance

    return scene_penetration


def compute_total_scene_penetrations(
    scenes: torch.Tensor,
    scene_vec_desc: SceneVecDescription,
    num_workers: int = 1,
) -> torch.Tensor:
    """
    Compute the total scene penetration as the sum of the penetration distances of all
    objects in the scene. A higher number means more penetration.

    Args:
        scenes (torch.Tensor): The unormalized scene tensors of shape (B, N, V).
        scene_vec_desc (SceneVecDescription): The scene vector description.
        num_workers (int): The number of workers to use for parallel processing.

    Returns:
        torch.Tensor: The total scene penetration for each scene in `scenes` of shape
            (B,).
    """
    scenes_np = scenes.detach().cpu().numpy()

    penetration_func = partial(
        compute_total_scene_penetration, scene_vec_desc=scene_vec_desc
    )

    if num_workers == 1 or len(scenes_np) == 1:
        total_penetrations = torch.tensor(
            [penetration_func(scene) for scene in scenes_np]
        )
    else:
        num_workers = min([num_workers, len(scenes_np), multiprocessing.cpu_count()])
        with multiprocessing.Pool(num_workers) as pool:
            total_penetrations = pool.map(penetration_func, scenes_np)
        total_penetrations = torch.tensor(total_penetrations)

    return total_penetrations


def compute_total_minimum_distance(
    scene: np.ndarray, scene_vec_desc: SceneVecDescription
) -> float:
    """
    Compute the total minimum distance as the sum of the minimum distances of all
    objects in the scene. A greater than zero minimum distance indicates floating
    objects that cannot be in static equilibrium.

    Args:
        scene (np.ndarray): The unormalized scene tensor. Shape (N, V) where N is the
            number of objects and V is the object feature vector length.
        scene_vec_desc (SceneVecDescription): The scene vector description.

    Returns:
        float: The total minimum distance.
    """
    # Obtain the plant and scene graph.
    builder = DiagramBuilder()
    result = create_plant_and_scene_graph_from_scene(
        scene=scene,
        builder=builder,
        scene_vec_desc=scene_vec_desc,
        weld_objects=False,
    )
    plant = result.plant
    scene_graph = result.scene_graph
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    scene_graph_context = scene_graph.GetMyContextFromRoot(context)
    query_object: QueryObject = scene_graph.get_query_output_port().Eval(
        scene_graph_context
    )
    inspector = query_object.inspector()

    # Get all signed distance pairs. Use a max distance to improve performance.
    try:
        signed_distance_pairs: List[
            SignedDistancePair
        ] = query_object.ComputeSignedDistancePairwiseClosestPoints(max_distance=5.0)
    except:
        # See https://github.com/RobotLocomotion/drake/issues/21673 for why this might
        # occur.
        print("Computing signed distance failed, returning zero total_min_distance.")
        return 0.0

    # Get the minimum distance for each object.
    obj_to_min_distance = {}
    for pair in signed_distance_pairs:
        frameA_id = inspector.GetFrameId(pair.id_A)
        frameB_id = inspector.GetFrameId(pair.id_B)
        modelA = ModelInstanceIndex(inspector.GetFrameGroup(frameA_id))
        modelB = ModelInstanceIndex(inspector.GetFrameGroup(frameB_id))
        if plant.num_positions(modelA) == 0 and plant.num_positions(modelB) == 0:
            # Two welded objects in penetration. Continue.
            continue

        # Count penetrations as zero distance.
        obj_to_min_distance[pair.id_A] = max(
            0.0, min(obj_to_min_distance.get(pair.id_A, float("inf")), pair.distance)
        )
        obj_to_min_distance[pair.id_B] = max(
            0.0, min(obj_to_min_distance.get(pair.id_B, float("inf")), pair.distance)
        )

    # Sum up the minimum distances for all objects.
    total_min_distance = sum(obj_to_min_distance.values())

    # Threshhold at 0.
    total_min_distance = max(0.0, total_min_distance)
    return total_min_distance


def compute_total_minimum_distances(
    scenes: torch.Tensor,
    scene_vec_desc: SceneVecDescription,
    num_workers: int = 1,
) -> torch.Tensor:
    """
    Compute the total minimum distance as the sum of the minimum distances of all
    objects in the scene. A greater than zero minimum distance indicates floating
    objects that cannot be in static equilibrium.

    Args:
        scenes (torch.Tensor): The unormalized scene tensors of shape (B, N, V).
        scene_vec_desc (SceneVecDescription): The scene vector description.
        num_workers (int): The number of workers to use for parallel processing.

    Returns:
        torch.Tensor: The total minimum distance for each scene in `scenes` of shape
            (B,).
    """
    scenes_np = scenes.detach().cpu().numpy()

    distance_func = partial(
        compute_total_minimum_distance, scene_vec_desc=scene_vec_desc
    )

    if num_workers == 1 or len(scenes_np) == 1:
        total_min_distances = torch.tensor(
            [distance_func(scene) for scene in scenes_np]
        )
    else:
        num_workers = min([num_workers, len(scenes_np), multiprocessing.cpu_count()])
        with multiprocessing.Pool(num_workers) as pool:
            total_min_distances = pool.map(distance_func, scenes_np)
        total_min_distances = torch.tensor(total_min_distances)

    return total_min_distances


def categorical_kl(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute the KL divergence between two categorical distributions p and q.

    Args:
        p (np.ndarray): The first categorical distribution.
        q (np.ndarray): The second categorical distribution.

    Returns:
        float: The KL divergence between the two distributions.
    """
    return (p * (np.log(p + 1e-6) - np.log(q + 1e-6))).sum()


def compute_scene_object_kl_divergence_metric(
    dataset_scenes: torch.Tensor,
    dataset_scene_vec_desc: SceneVecDescription,
    synthesized_scenes: torch.Tensor,
    synthesized_scene_vec_desc: SceneVecDescription,
) -> float:
    """
    Compute the object category KL divergence metric between the dataset scenes and
    generated scenes. The metric computes the categorical KL divergence between the
    categorical object distributions over the entire dataset (not per scene).

    Args:
        dataset_scenes (torch.Tensor): The unnormalized dataset scenes tensor of shape
            (B, N, V).
        dataset_scene_vec_desc (SceneVecDescription): The scene vector description for
            the dataset scenes.
        synthesized_scenes (torch.Tensor): The unnormalized synthesized scenes tensor
            of shape (B1, N, V).
        synthesized_scene_vec_desc (SceneVecDescription): The scene vector description
            for the synthesized scenes.

    Returns:
        float: The object category KL divergence metric.
    """
    # Count the dataset objects.
    dataset_objs = dataset_scenes.reshape(
        (-1, dataset_scenes.shape[-1])
    )  # Shape (B*N, V)
    dataset_obj_counts = {}
    for obj in dataset_objs:
        model_path = dataset_scene_vec_desc.get_model_path(obj)
        if model_path is None:
            continue
        elif model_path in dataset_obj_counts:
            dataset_obj_counts[model_path] += 1
        else:
            dataset_obj_counts[model_path] = 1

    # Count the synthesized objects.
    synthesized_objs = synthesized_scenes.reshape(
        (-1, synthesized_scenes.shape[-1])
    )  # Shape (B1*N, V)
    synthesized_obj_counts = {}
    for obj in synthesized_objs:
        model_path = synthesized_scene_vec_desc.get_model_path(obj)
        if model_path is None:
            continue
        elif model_path in synthesized_obj_counts:
            synthesized_obj_counts[model_path] += 1
        else:
            synthesized_obj_counts[model_path] = 1

    # Construct the distributions by normalizing.
    dataset_obj_count = sum(dataset_obj_counts.values())
    normalized_dataset_obj_counts = {
        path: count / dataset_obj_count for path, count in dataset_obj_counts.items()
    }
    synthesized_obj_count = sum(synthesized_obj_counts.values())
    normalized_synthesized_obj_counts = {
        path: count / synthesized_obj_count
        for path, count in synthesized_obj_counts.items()
    }

    # Convert to lists of same length.
    dataset_distribution = []
    synthesized_distribution = []
    for model_path in set(
        dataset_scene_vec_desc.model_paths + synthesized_scene_vec_desc.model_paths
    ):
        dataset_distribution.append(normalized_dataset_obj_counts.get(model_path, 0.0))
        synthesized_distribution.append(
            normalized_synthesized_obj_counts.get(model_path, 0.0)
        )
    assert 0.9999 <= sum(dataset_distribution) <= 1.0001
    assert 0.9999 <= sum(synthesized_distribution) <= 1.0001

    # Compute the KL divergence.
    kl_divergence = categorical_kl(
        np.asarray(dataset_distribution), np.asanyarray(synthesized_distribution)
    )

    return kl_divergence


def compute_welded_object_pose_deviation_metric(
    dataset_scenes: torch.Tensor,
    dataset_scene_vec_desc: SceneVecDescription,
    synthesized_scenes: torch.Tensor,
    synthesized_scene_vec_desc: SceneVecDescription,
) -> float:
    """
    Compute the welded object pose deviation metric between the dataset scenes and
    synthesized scenes. The metric computes the mean pose deviation of all welded
    objects in the scenes.

    Args:
        dataset_scenes (torch.Tensor): The unnormalized dataset scenes tensor of
            shape (B, N, V).
        dataset_scene_vec_desc (SceneVecDescription): The scene vector description for
            the dataset scenes.
        synthesized_scenes (torch.Tensor): The unnormalized synthesized scenes tensor
            of shape (B1, N, V).
        synthesized_scene_vec_desc (SceneVecDescription): The scene vector description
            for the synthesized scenes.

    Returns:
        float: The welded object pose deviation metric. Zero is best.
    """
    model_paths = list(
        set(dataset_scene_vec_desc.welded_object_model_paths)
        | set(synthesized_scene_vec_desc.welded_object_model_paths)
    )
    num_welded_objects = len(model_paths)
    if num_welded_objects == 0:
        return 0.0

    # Reshape scenes to (B*N, V).
    dataset_objects = dataset_scenes.reshape(-1, dataset_scenes.shape[-1])
    synthesized_objects = synthesized_scenes.reshape(-1, synthesized_scenes.shape[-1])

    # Get the welded object indices in the model paths list.
    welded_object_indices = [
        dataset_scene_vec_desc.model_paths.index(model_path)
        for model_path in model_paths
    ]
    synthesized_welded_object_indices = [
        synthesized_scene_vec_desc.model_paths.index(model_path)
        for model_path in model_paths
    ]

    total_pose_deviation = 0.0
    for i in range(num_welded_objects):
        # Get dataset object poses for this welded object type.
        dataset_model_path_vec = dataset_scene_vec_desc.get_model_path_vec(
            dataset_objects
        )
        dataset_indices = dataset_model_path_vec[:, welded_object_indices[i]].nonzero(
            as_tuple=True
        )[0]

        # Get synthesized object poses for this welded object type.
        synth_model_path_vec = synthesized_scene_vec_desc.get_model_path_vec(
            synthesized_objects
        )
        synth_indices = synth_model_path_vec[
            :, synthesized_welded_object_indices[i]
        ].nonzero(as_tuple=True)[0]

        if len(dataset_indices) == 0 and len(synth_indices) == 0:
            # Welded object appears in neither dataset nor synthesized.
            continue
        elif len(dataset_indices) == 0 or len(synth_indices) == 0:
            # Welded object appears in one but not the other.
            total_pose_deviation += float("inf")
            continue

        # Get dataset object poses.
        dataset_objects_of_type = dataset_objects[dataset_indices]
        dataset_poses = dataset_scene_vec_desc.get_scene_without_model_path(
            dataset_objects_of_type
        )
        dataset_mean_pose = torch.mean(dataset_poses, dim=0)

        # Get synthesized object poses.
        synth_objects_of_type = synthesized_objects[synth_indices]
        synth_poses = synthesized_scene_vec_desc.get_scene_without_model_path(
            synth_objects_of_type
        )
        synth_mean_pose = torch.mean(synth_poses, dim=0)

        # Compute L2 distance between mean poses.
        pose_deviation = torch.norm(dataset_mean_pose - synth_mean_pose).item()
        total_pose_deviation += pose_deviation

    # Average over all welded objects.
    mean_pose_deviation = total_pose_deviation / num_welded_objects
    return mean_pose_deviation
