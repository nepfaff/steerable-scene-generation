import os

from functools import partial
from multiprocessing import Pool
from typing import Any, Callable

import torch

from omegaconf import DictConfig
from tqdm import tqdm

from steerable_scene_generation.algorithms.common.dataclasses import SceneVecDescription
from steerable_scene_generation.algorithms.scene_diffusion.inpainting_helpers import (
    generate_empty_object_inpainting_masks,
    generate_non_penetration_inpainting_masks,
    generate_physical_feasibility_inpainting_masks,
)
from steerable_scene_generation.algorithms.scene_diffusion.mcts_helpers import (
    MCTSNode,
    backpropagate,
    compute_reward,
    find_best_nodes,
    remove_duplicate_nodes,
    select_node_with_uct,
)
from steerable_scene_generation.datasets.scene.scene import SceneDataset


def generate_inference_time_search_mask_and_metadata(
    scene: torch.Tensor,
    cfg: DictConfig,
    scene_vec_desc: SceneVecDescription,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Generate an inpainting mask and metadata for a scene based on the selected
    objective.

    Args:
        scene (torch.Tensor): The scene tensor of shape (B, N, V). This is
            unnormalized.
        cfg (DictConfig): The configuration for the inference time search.
        scene_vec_desc (SceneVecDescription): The scene vector descriptor.

    Returns:
        tuple[torch.Tensor, dict[str, Any]]: A tuple containing:
            - inpainting_mask: Boolean tensor of shape (B, N, V).
            - metadata: Dictionary with objective-specific metrics.
    """
    num_objectives = sum(
        [
            cfg.use_non_penetration_objective,
            cfg.use_physical_feasibility_objective,
            cfg.use_object_number_objective,
        ]
    )
    if num_objectives == 0 or num_objectives > 1:
        raise ValueError("Exactly one objective must be used.")

    metadata = {}

    if cfg.use_non_penetration_objective:
        (
            inpainting_mask,
            total_penetration_distance,
        ) = generate_non_penetration_inpainting_masks(
            scenes=scene,
            scene_vec_desc=scene_vec_desc,
            threshold=cfg.non_penetration.threshold,
            exclude_welded_objects=True,
        )
        metadata["total_penetration_distance"] = total_penetration_distance[0]

    elif cfg.use_physical_feasibility_objective:
        (
            inpainting_mask,
            total_penetration_distance,
            non_static_equilibrium_object_number,
        ) = generate_physical_feasibility_inpainting_masks(
            scenes=scene,
            scene_vec_desc=scene_vec_desc,
            non_penetration_threshold=cfg.non_penetration.threshold,
            use_sim=cfg.physical_feasibility.use_sim,
            sim_duration=cfg.physical_feasibility.sim_duration,
            sim_time_step=cfg.physical_feasibility.sim_time_step,
            sim_translation_threshold=cfg.physical_feasibility.sim_translation_threshold,
            sim_rotation_threshold=cfg.physical_feasibility.sim_rotation_threshold,
            static_equilibrium_distance_threshold=cfg.physical_feasibility.static_equilibrium_distance_threshold,
            exclude_welded_objects_from_non_penetration_masking=cfg.physical_feasibility.exclude_welded_objects_from_non_penetration_masking,
        )
        metadata["total_penetration_distance"] = total_penetration_distance[0]
        metadata[
            "non_static_equilibrium_object_number"
        ] = non_static_equilibrium_object_number[0]

    elif cfg.use_object_number_objective:
        (
            physical_mask,
            total_penetration_distance,
            non_static_equilibrium_object_number,
        ) = generate_physical_feasibility_inpainting_masks(
            scenes=scene,
            scene_vec_desc=scene_vec_desc,
            non_penetration_threshold=cfg.non_penetration.threshold,
            use_sim=cfg.physical_feasibility.use_sim,
            sim_duration=cfg.physical_feasibility.sim_duration,
            sim_time_step=cfg.physical_feasibility.sim_time_step,
            sim_translation_threshold=cfg.physical_feasibility.sim_translation_threshold,
            sim_rotation_threshold=cfg.physical_feasibility.sim_rotation_threshold,
            static_equilibrium_distance_threshold=cfg.physical_feasibility.static_equilibrium_distance_threshold,
            exclude_welded_objects_from_non_penetration_masking=cfg.physical_feasibility.exclude_welded_objects_from_non_penetration_masking,
        )

        empty_mask, empty_object_numbers = generate_empty_object_inpainting_masks(
            scenes=scene, scene_vec_desc=scene_vec_desc
        )

        # Combine masks
        inpainting_mask = torch.logical_or(physical_mask, empty_mask)

        metadata["total_penetration_distance"] = total_penetration_distance[0]
        metadata[
            "non_static_equilibrium_object_number"
        ] = non_static_equilibrium_object_number[0]
        metadata["empty_object_number"] = empty_object_numbers[0]
        max_num_objects = scene.shape[1]
        metadata["non_empty_object_number"] = max_num_objects - empty_object_numbers[0]

    return inpainting_mask, metadata


def update_inference_time_search_additional_info(
    cfg: DictConfig,
    additional_info: dict[str, list[Any]],
    metadata: dict[str, Any],
) -> dict[str, list[Any]]:
    """
    Update the additional info dictionary with new metadata.

    Args:
        cfg (DictConfig): The configuration for the inference time search.
        additional_info (dict[str, list[Any]]): The existing additional info
            dictionary.
        metadata (dict[str, Any]): The new metadata to add.

    Returns:
        dict[str, list[Any]]: The updated additional info dictionary.
    """
    # Initialize additional_info if it's empty
    if not additional_info:
        additional_info = {}

    # Update based on the objective
    if cfg.use_non_penetration_objective:
        if "total_penetration_distances" not in additional_info:
            additional_info["total_penetration_distances"] = []
        additional_info["total_penetration_distances"].append(
            metadata["total_penetration_distance"]
        )

    elif cfg.use_physical_feasibility_objective:
        if "total_penetration_distances" not in additional_info:
            additional_info["total_penetration_distances"] = []
        if "non_static_equilibrium_object_numbers" not in additional_info:
            additional_info["non_static_equilibrium_object_numbers"] = []

        additional_info["total_penetration_distances"].append(
            metadata["total_penetration_distance"]
        )
        additional_info["non_static_equilibrium_object_numbers"].append(
            metadata["non_static_equilibrium_object_number"]
        )

    elif cfg.use_object_number_objective:
        if "total_penetration_distances" not in additional_info:
            additional_info["total_penetration_distances"] = []
        if "non_static_equilibrium_object_numbers" not in additional_info:
            additional_info["non_static_equilibrium_object_numbers"] = []
        if "empty_object_numbers" not in additional_info:
            additional_info["empty_object_numbers"] = []
        if "non_empty_object_numbers" not in additional_info:
            additional_info["non_empty_object_numbers"] = []

        additional_info["total_penetration_distances"].append(
            metadata["total_penetration_distance"]
        )
        additional_info["non_static_equilibrium_object_numbers"].append(
            metadata["non_static_equilibrium_object_number"]
        )
        additional_info["empty_object_numbers"].append(metadata["empty_object_number"])
        additional_info["non_empty_object_numbers"].append(
            metadata["non_empty_object_number"]
        )

    return additional_info


def process_child_scene(
    i: int,
    child_scenes: torch.Tensor,
    node: MCTSNode,
    cfg: DictConfig,
    scene_vec_desc: SceneVecDescription,
) -> dict[str, Any] | None:
    """
    Process a single child scene in the MCTS expansion phase.

    Args:
        i: Index of the child scene.
        child_scenes: Tensor containing all child scenes.
        node: Parent MCTS node.
        cfg: Configuration.
        scene_vec_desc: Scene vector descriptor.

    Returns:
        Dictionary with processed child scene data or None if the child should be
        skipped.
    """
    child_scene = child_scenes[i : i + 1]  # Shape (1, N, V)

    # Generate new inpainting mask and metadata.
    child_mask, child_metadata = generate_inference_time_search_mask_and_metadata(
        scene=child_scene, cfg=cfg, scene_vec_desc=scene_vec_desc
    )

    # Check if we should consider this child.
    if cfg.mcts.only_consider_children_with_different_mask and torch.all(
        child_mask == node.inpainting_mask
    ):
        return None

    # Calculate cost.
    cost = child_mask.sum().item()

    return {
        "scene": child_scene,
        "mask": child_mask,
        "metadata": child_metadata,
        "cost": cost,
    }


def mcts_inference_time_search(
    inpaint_data_batch: dict[str, torch.Tensor],
    max_num_objects_per_scene: int,
    use_ema: bool,
    cfg: DictConfig,
    scene_vec_desc: SceneVecDescription,
    dataset: SceneDataset,
    inpaint_function: Callable,
) -> dict[str, Any]:
    """
    Samples a single scene using MCTS inference-time search.
    See `inference_time_search` for more details.

    Args:
        inpaint_data_batch: A dictionary containing the input data for inpainting.
        max_num_objects_per_scene: The maximum number of objects per scene.
        use_ema: Whether to use exponential moving average for inpainting.
        cfg: The configuration for the inference-time search.
        scene_vec_desc: The scene vector descriptor.
        dataset: The dataset to use for inpainting.
        inpaint_function: The function to inpaint a scene.

    Returns:
        A dictionary containing the results of the inference-time search.
    """
    # Create root node. Starting the root node as all masked out makes it less likely
    # to get stuck in a bad local minimum at the start due to a bad noise sample.
    scene_shape = (
        1,
        max_num_objects_per_scene,
        scene_vec_desc.get_object_vec_len(),
    )  # Shape (1, N, V)
    device = inpaint_data_batch["scenes"].device
    initial_mask = torch.ones(scene_shape, dtype=torch.bool)
    root = MCTSNode(
        scene=torch.zeros(scene_shape, dtype=inpaint_data_batch["scenes"].dtype),
        inpainting_mask=initial_mask,
        metadata={},
    )

    # Track history for visualization.
    history: list[torch.Tensor] = []

    # Track best cost during search for logging.
    best_cost: float = initial_mask.sum().item()
    best_scene_indices: list[int] = []
    best_costs: list[float] = []

    # Initialize additional info.
    additional_info: dict[str, list[Any]] = {}

    # MCTS main loop.
    max_iterations: int = cfg.max_steps
    branching_factor: int = cfg.mcts.branching_factor
    num_processes = min(os.cpu_count(), branching_factor)
    iteration: int = 0

    for iteration in tqdm(range(max_iterations), desc="MCTS Search", position=0):
        # --- SELECTION ---
        node = root
        # Descend until we find a leaf or terminal node.
        while not node.is_leaf() and not node.is_terminal():
            node = select_node_with_uct(
                node=node,
                exploration_weight=cfg.mcts.exploration_weight,
            )

        # --- EXPANSION ---
        if node.is_terminal():
            # Found a perfect solution.
            break
        else:
            # Create a data batch for inpainting.
            normalized_scene = dataset.normalize_scenes(node.scene)

            # Replicate the scene and mask for batch processing.
            batched_scenes = normalized_scene.repeat(
                branching_factor, 1, 1
            )  # Shape (B, N, V)
            batched_masks = node.inpainting_mask.repeat(
                branching_factor, 1, 1
            )  # Shape (B, N, V)

            # Inpaint all scenes to get the children.
            inpaint_data_batch["scenes"] = batched_scenes.to(device)
            inpaint_data_batch["inpainting_masks"] = batched_masks.to(device)
            child_scenes: torch.Tensor = inpaint_function(
                inpaint_data_batch, use_ema=use_ema
            )  # Shape (B, N, V)

            # Move tensors to CPU before passing to multiprocessing.
            child_scenes = child_scenes.cpu()

            # Process child scenes in parallel.
            with Pool(processes=num_processes) as pool:
                results: list[dict[str, Any] | None] = list(
                    tqdm(
                        pool.imap(
                            partial(
                                process_child_scene,
                                child_scenes=child_scenes,
                                node=node,
                                cfg=cfg,
                                scene_vec_desc=scene_vec_desc,
                            ),
                            range(branching_factor),
                        ),
                        total=branching_factor,
                        desc=" Expanding",
                        leave=False,
                        position=1,
                    )
                )

            # Process results and create child nodes.
            for result in results:
                if result is None:
                    continue

                child_scene: torch.Tensor = result["scene"]
                child_mask: torch.Tensor = result["mask"]
                child_metadata: dict[str, Any] = result["metadata"]
                cost: float = result["cost"]

                # Add to history.
                history.append(child_scene)
                history_index: int = len(history) - 1

                # Update additional info.
                additional_info = update_inference_time_search_additional_info(
                    cfg=cfg, additional_info=additional_info, metadata=child_metadata
                )

                # Create child node.
                child_node = MCTSNode(
                    scene=child_scene,
                    inpainting_mask=child_mask,
                    parent=node,
                    metadata=child_metadata,
                )
                node.children.append(child_node)

                # Check if this is the best scene so far.
                if cost < best_cost:
                    best_cost = cost
                    best_scene_indices.append(history_index)
                    best_costs.append(cost)

            # The node didn't produce any children.
            if len(node.children) == 0:
                continue

            # Select the first child for rollout.
            node = node.children[0]

        # --- ROLLOUT ---
        # For simplicity, we use the immediate reward without additional simulation.
        value = compute_reward(node)

        # --- BACKPROPAGATION ---
        backpropagate(node=node, value=value)

    # Find all best nodes.
    best_nodes = find_best_nodes(root)

    # Remove duplicates.
    best_nodes = remove_duplicate_nodes(best_nodes)

    best_scenes: list[torch.Tensor] = []
    for best_node in best_nodes:
        # Process the best scene by removing problematic objects.
        object_level_masks = best_node.inpainting_mask.any(dim=2)  # Shape (1, N)
        processed_scene = scene_vec_desc.replace_masked_objects_with_empty(
            scene=best_node.scene, mask=object_level_masks
        )
        best_scenes.append(processed_scene)

    # Get tree data for visualization.
    tree_data = root.get_tree_data()

    # Return results.
    return {
        "reached_max_iters": iteration == max_iterations - 1,
        "num_iters_used": iteration + 1,
        "history": torch.cat(history, dim=0).squeeze(1),  # Shape (T, N, V)
        "best_scenes": torch.cat(best_scenes, dim=0),  # Shape (K, N, V)
        "best_scene_indices": best_scene_indices,
        "best_costs": best_costs,
        "tree_data": tree_data,
        **additional_info,
    }
