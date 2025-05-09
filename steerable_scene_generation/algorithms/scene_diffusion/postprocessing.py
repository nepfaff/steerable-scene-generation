import logging
import multiprocessing
import time

from datetime import timedelta
from functools import partial
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from pydrake.all import (
    Context,
    EventStatus,
    InverseKinematics,
    IpoptSolver,
    Simulator,
    SnoptSolver,
    SolverOptions,
)

from steerable_scene_generation.algorithms.common.dataclasses import (
    PlantSceneGraphCache,
    SceneVecDescription,
)
from steerable_scene_generation.utils.drake_utils import (
    create_plant_and_scene_graph_from_scene_with_cache,
    update_scene_poses_from_plant,
)

logger = logging.getLogger(__name__)


def apply_non_penetration_projection_single_scene(
    scene: torch.Tensor,
    cache: Union[PlantSceneGraphCache, None],
    scene_vec_desc: SceneVecDescription,
    translation_only: bool,
    influence_distance: float = 0.02,
    solver_name: str = "snopt",
    iteration_limit: int = 5000,
    return_cache: bool = True,
) -> Tuple[torch.Tensor, Union[PlantSceneGraphCache, None], bool]:
    """See `apply_non_penetration_projection` for more details."""
    # Obtain the plant and scene graph.
    cache, _, plant_context = create_plant_and_scene_graph_from_scene_with_cache(
        scene=scene,
        scene_vec_desc=scene_vec_desc,
        cache=cache,
    )
    plant = cache.plant

    # Set up projection NLP.
    ik = InverseKinematics(plant, plant_context)
    q_vars = ik.q()
    prog = ik.prog()

    # Stay close to initial positions.
    q0 = plant.GetPositions(plant_context)
    if len(q0) == 0:
        logging.warning("No DOFs found for plant. Skipping non-penetration projection.")
        return scene, cache, False

    for body_idx, model_idx in zip(cache.rigid_body_indices, cache.model_indices):
        if body_idx is None:
            continue

        body = plant.get_body(body_idx)
        if not body.is_floating():
            # Skip non-floating bodies.
            continue

        # For two quaternion z and z0 with angle θ between their orientation,
        # we know
        # 1-cosθ = 2 - 2*(zᵀz₀)² = 2 - 2zᵀz₀z₀ᵀz
        # So we can add a quadratic cost on the quaternion z.
        q_start_idx = body.floating_positions_start()
        model_quat_vars = q_vars[q_start_idx : q_start_idx + 4]
        z0 = q0[q_start_idx : q_start_idx + 4]
        prog.AddQuadraticCost(
            -4 * np.outer(z0, z0), np.zeros((4,)), 2, model_quat_vars, is_convex=False
        )
        model_pos_vars = q_vars[q_start_idx + 4 : q_start_idx + 7]
        prog.AddQuadraticErrorCost(
            np.eye(3), q0[q_start_idx + 4 : q_start_idx + 7], model_pos_vars
        )

    # Nonpenetration constraint.
    ik.AddMinimumDistanceLowerBoundConstraint(0.0, influence_distance)

    if translation_only:
        # Add constraints for rotations to stay constant.
        for body_idx, model_idx in zip(cache.rigid_body_indices, cache.model_indices):
            if body_idx is None:
                # Skip empty objects.
                continue

            body = plant.get_body(body_idx)
            if not body.is_floating():
                # Skip non-floating bodies.
                continue

            # Get rotation decision variables.
            q_start_idx = body.floating_positions_start()
            model_quat_vars = q_vars[q_start_idx : q_start_idx + 4]

            # Get initial rotation.
            model_q = plant.GetPositions(plant_context, model_idx)
            model_quat = model_q[:4]

            # Add constraint for rotation to stay constant.
            prog.AddBoundingBoxConstraint(
                model_quat,  # lb
                model_quat,  # ub
                model_quat_vars,  # vars
            )

    # Use the starting positions as the initial guess.
    prog.SetInitialGuess(q_vars, q0)

    # Solve.
    options = SolverOptions()
    if solver_name == "snopt":
        solver = SnoptSolver()
        options.SetOption(solver.id(), "Major feasibility tolerance", 1e-3)
        options.SetOption(solver.id(), "Major optimality tolerance", 1e-3)
        options.SetOption(solver.id(), "Major iterations limit", iteration_limit)
        options.SetOption(solver.id(), "Time limit", 60)
        options.SetOption(solver.id(), "Timing level", 3)
    elif solver_name == "ipopt":
        solver = IpoptSolver()
        options.SetOption(solver.id(), "max_iter", iteration_limit)
    else:
        raise ValueError(f"Invalid solver: {solver_name}")
    if not solver.available():
        raise ValueError(f"Solver {solver_name} is not available.")

    try:
        result = solver.Solve(prog, None, options)
        success = result.is_success()

        # Update the scene poses.
        plant.SetPositions(plant_context, result.GetSolution(q_vars))
        projected_scene = update_scene_poses_from_plant(
            scene=scene,
            plant=plant,
            plant_context=plant_context,
            model_indices=cache.model_indices,
            scene_vec_desc=scene_vec_desc,
        )
    except Exception as e:
        logger.warning(
            "Projection failed with exception. Returning original scene. "
            + f"Exception: {e}"
        )
        projected_scene = scene
        success = False

    if not success:
        logger.warning("Projection failed.")

    return projected_scene, (cache if return_cache else None), success


def apply_non_penetration_projection(
    scenes: torch.Tensor,
    scene_vec_desc: SceneVecDescription,
    translation_only: bool,
    influence_distance: float,
    solver_name: str,
    caches: Union[List[PlantSceneGraphCache], List[None]],
    iteration_limit: int = 5000,
    num_workers: int = 1,
) -> Tuple[torch.Tensor, Union[List[PlantSceneGraphCache], List[None]], List[bool]]:
    """
    Apply non-penetration projection to a batch of scenes.

    Args:
        scenes (torch.Tensor): Input scenes of shape (B, N, V) where B is the batch
            size, N is the number of objects, and V is the object feature vector
            length.
        scene_vec_desc (SceneVecDescription): Scene vector description.
        translation_only (bool): Whether to only optimize translations.
        influence_distance (float): Influence distance in meters for non-penetration
            constraint. Try to increase this if the projection fails.
        solver_name (str): Name of the solver to use. Either "snopt" or "ipopt".
        caches (List[PlantSceneGraphCache]): List of caches.
        iteration_limit (int): Maximum number of iterations for the solver.
        num_workers (int, optional): Number of workers to use.

    Returns:
        Tuple[torch.Tensor, Union[List[PlantSceneGraphCache], List[None]], List[bool]]:
            A tuple containing:
                - The projected scenes.
                - The updated caches.
                - A list of projection success flags for each scene.
    """
    assert num_workers > 0
    assert len(scenes) == len(caches)

    start_time = time.time()

    scenes = scenes.cpu().detach()

    if num_workers == 1 or len(scenes) == 1:
        projected_scenes, new_caches, successes = [], [], []
        for scene, cache in zip(scenes, caches):
            (
                projected_scene,
                cache,
                success,
            ) = apply_non_penetration_projection_single_scene(
                scene=scene,
                cache=cache,
                scene_vec_desc=scene_vec_desc,
                translation_only=translation_only,
                influence_distance=influence_distance,
                solver_name=solver_name,
                iteration_limit=iteration_limit,
            )
            projected_scenes.append(projected_scene)
            new_caches.append(cache)
            successes.append(success)
    else:
        num_workers = min(num_workers, len(scenes), multiprocessing.cpu_count())
        with multiprocessing.Pool(num_workers) as pool:
            result = pool.starmap(
                partial(
                    apply_non_penetration_projection_single_scene,
                    scene_vec_desc=scene_vec_desc,
                    translation_only=translation_only,
                    solver_name=solver_name,
                    iteration_limit=iteration_limit,
                    return_cache=False,  # Can't return non-pickeable objects.
                ),
                zip(scenes, caches),
            )
            projected_scenes, _, successes = zip(*result)
            successes = list(successes)

        # Caches stay the same as the objects in the scene aren't changed by the
        # projection.
        new_caches = caches

    logger.info(
        f"Projecting {len(scenes)} scenes took {timedelta(seconds=time.time()-start_time)}"
    )

    return torch.stack(projected_scenes), new_caches, successes


def apply_forward_simulation_single_scene(
    scene: torch.Tensor,
    cache: Union[PlantSceneGraphCache, None],
    scene_vec_desc: SceneVecDescription,
    simulation_time_s: float,
    time_step: float,
    timeout_s: Optional[float] = None,
    return_cache: bool = True,
) -> Tuple[torch.Tensor, Union[PlantSceneGraphCache, None]]:
    """See `apply_forward_simulation` for more details."""
    # Obtain the diagram and plant.
    cache, context, plant_context = create_plant_and_scene_graph_from_scene_with_cache(
        scene=scene,
        scene_vec_desc=scene_vec_desc,
        time_step=time_step,
        cache=cache,
    )

    start_time = time.time()

    def timeout_monitor(context: Context) -> EventStatus:
        if time.time() - start_time > timeout_s:
            return EventStatus.ReachedTermination(None, "timeout")
        return EventStatus.DidNothing()

    # Simulate.
    simulator = Simulator(cache.diagram, context)
    if timeout_s is not None:
        simulator.set_monitor(timeout_monitor)
    simulator.AdvanceTo(simulation_time_s)

    # Update the scene poses.
    simulated_scene = update_scene_poses_from_plant(
        scene=scene,
        plant=cache.plant,
        plant_context=plant_context,
        model_indices=cache.model_indices,
        scene_vec_desc=scene_vec_desc,
    )

    return simulated_scene, (cache if return_cache else None)


def apply_forward_simulation(
    scenes: torch.Tensor,
    scene_vec_desc: SceneVecDescription,
    simulation_time_s: float,
    time_step: float,
    caches: Union[List[PlantSceneGraphCache], List[None]],
    timeout_s: Optional[float] = None,
    num_workers: int = 1,
) -> Tuple[torch.Tensor, Union[List[PlantSceneGraphCache], List[None]]]:
    """
    Apply forward simulation to a batch of scenes. This is useful for ensuring that the
    scenes are in static equilibrium.

    Args:
        scenes (torch.Tensor): Input scenes of shape (B, N, V) where B is the batch
            size, N is the number of objects, and V is the object feature vector
            length.
        scene_vec_desc (SceneVecDescription): Scene vector description.
        simulation_time_s (float): Simulation time in seconds.
        time_step (float): Time step for simulation.
        caches (List[PlantSceneGraphCache]): List of caches.
        timeout_s (Optional[float], optional): Timeout in seconds for each simulation.
        num_workers (int, optional): Number of workers to use.

    Returns:
        Tuple[torch.Tensor, Union[List[PlantSceneGraphCache], List[None]]]:
            A tuple containing:
                - The simulated scenes.
                - The updated caches.
    """
    assert num_workers > 0
    assert len(scenes) == len(
        caches
    ), f"Got {len(scenes)} scenes and {len(caches)} caches"

    start_time = time.time()

    scenes = scenes.cpu().detach()

    if num_workers == 1 or len(scenes) == 1:
        simulated_scenes, new_caches = [], []
        for scene, cache in zip(scenes, caches):
            simulated_scene, cache = apply_forward_simulation_single_scene(
                scene=scene,
                cache=cache,
                scene_vec_desc=scene_vec_desc,
                simulation_time_s=simulation_time_s,
                time_step=time_step,
                timeout_s=timeout_s,
            )
            simulated_scenes.append(simulated_scene)
            new_caches.append(cache)
    else:
        num_workers = min(num_workers, len(scenes), multiprocessing.cpu_count())
        with multiprocessing.Pool(num_workers) as pool:
            result = pool.starmap(
                partial(
                    apply_forward_simulation_single_scene,
                    scene_vec_desc=scene_vec_desc,
                    simulation_time_s=simulation_time_s,
                    time_step=time_step,
                    timeout_s=timeout_s,
                    return_cache=False,  # Can't return non-pickeable objects.
                ),
                zip(scenes, caches),
            )
            simulated_scenes, _ = zip(*result)

        # Caches stay the same as the objects in the scene aren't changed by simulation.
        new_caches = caches

    logger.info(
        f"Simulating {len(scenes)} scenes took {timedelta(seconds=time.time()-start_time)}"
    )

    return torch.stack(simulated_scenes), new_caches
