import logging
import math

import torch
import wandb

from omegaconf import DictConfig
from tqdm import tqdm

from steerable_scene_generation.algorithms.common.dataclasses import SceneVecDescription
from steerable_scene_generation.utils.scene_language_annotation import (
    check_object_name_format,
    check_object_number_format,
    check_spatial_relation_format,
    extract_object_counts_from_prompt,
    extract_object_names,
    get_object_number,
)
from steerable_scene_generation.utils.visualization import get_scene_renders

console_logger = logging.getLogger(__name__)


def _evaluate_object_number_prompt(
    scene_vec_desc: SceneVecDescription, scene: torch.Tensor, gt_obj_num: int
) -> tuple[float, bool]:
    """
    Evaluate how well a scene follows an object number prompt.

    Args:
        scene_vec_desc: The scene vector description.
        scene: The scene tensor.
        gt_obj_num: The ground truth number of objects.

    Returns:
        tuple[float, bool]: A tuple containing:
            - score: The continuous score for how well the scene follows the prompt.
            - is_satisfied: Whether the scene exactly satisfies the prompt.
    """
    object_model_paths = [scene_vec_desc.get_model_path(obj) for obj in scene]
    obj_num = get_object_number(object_model_paths)
    diff = abs(obj_num - gt_obj_num)

    # Calculate continuous score.
    if gt_obj_num > 0:
        # Log-based penalty for over/underestimation.
        score = max(0.0, 1.0 - (math.log(1 + diff) / math.log(1 + gt_obj_num)))
    else:
        # Log-based penalty for false positives when gt_obj_num = 0.
        K = 10  # Scaling factor
        score = max(0.0, 1.0 - (math.log(1 + obj_num) / math.log(1 + K)))

    # Binary satisfaction - only true if exact match.
    is_satisfied = obj_num == gt_obj_num

    return score, is_satisfied


def _evaluate_object_name_prompt(
    scene_vec_desc: SceneVecDescription,
    prompt: str,
    scene: torch.Tensor,
    is_subset: bool,
) -> tuple[float, bool]:
    """
    Evaluate how well a scene follows an object name prompt.

    Args:
        scene_vec_desc: The scene vector description.
        prompt: The textual prompt describing the scene.
        scene: The scene tensor.
        is_subset: Whether this is a subset prompt.

    Returns:
        tuple[float, bool]: A tuple containing:
            - score: The continuous score for how well the scene follows the prompt.
            - is_satisfied: Whether the scene exactly satisfies the prompt.
    """
    object_model_paths = [scene_vec_desc.get_model_path(obj) for obj in scene]
    object_counts = extract_object_counts_from_prompt(prompt)
    if object_counts is None:
        console_logger.error(f"Prompt {prompt} is not an object name prompt.")
        # Return 0 score and False match.
        return 0.0, False

    # Get actual objects in scene.
    scene_objects = extract_object_names(object_model_paths)

    # Count objects in scene.
    scene_counts = {}
    for obj in scene_objects:
        scene_counts[obj] = scene_counts.get(obj, 0) + 1

    # Compare counts.
    total_correct = 0.0
    total_weight = 0.0
    exact_match = True
    for obj, gt_count in object_counts.items():
        pred_count = scene_counts.get(obj, 0)
        diff = abs(pred_count - gt_count)

        if is_subset:
            # Only check if at least `gt_count` are present (no penalty for
            # over-prediction).
            if pred_count >= gt_count:
                score = 1.0
            else:
                score = max(0.0, 1.0 - (diff / gt_count))
                exact_match = False
        else:
            score = max(0.0, 1.0 - (diff / gt_count))
            if pred_count != gt_count:
                exact_match = False

        total_correct += score * gt_count
        total_weight += gt_count

    if not is_subset:
        # Penalize extra predicted objects that are not in the ground truth.
        extra_objects = set(scene_counts.keys()) - set(object_counts.keys())
        if extra_objects:
            exact_match = False

        total_gt_count = max(1, sum(object_counts.values()))
        for obj in extra_objects:
            extra_count = scene_counts[obj]
            penalty = extra_count / total_gt_count
            total_correct -= penalty

    # Ensure total_correct doesn't go negative.
    total_correct = max(0.0, total_correct)

    # Calculate per-prompt score.
    prompt_score = total_correct / max(1, total_weight)

    return prompt_score, exact_match


def compute_prompt_following_metrics(
    scene_vec_desc: SceneVecDescription,
    prompts: list[str],
    scenes: torch.Tensor,
    disable_tqdm: bool = False,
) -> dict[str, float | list[float]]:
    """
    Compute metrics for how well scenes follow their prompts.

    Currently handles two types of prompts:
    1. Object number prompts: "A scene with {num_objects} objects."
    2. Object name prompts: "A scene with {count} {object}s and a {object}..."

    Metrics calculation:
    - Object number metric: For a prompt specifying n objects, the score is:
      * When n > 0: score = max(0, 1 - log(1 + |pred - n|) / log(1 + n))
      * When n = 0: score = max(0, 1 - log(1 + pred) / log(1 + K)), where K=10
      This provides a logarithmic penalty for incorrect object counts.

    - Object name metric: For prompts specifying object types and counts:
      * For each object type i with count n_i: score_i = max(0, 1 - |pred_i - n_i| / n_i)
      * For subset prompts: no penalty for over-prediction if pred_i ≥ n_i
      * For non-subset prompts: penalty for extra object types not in prompt
      * Final score = (sum(score_i * n_i) - penalties) / sum(n_i)

    Args:
        scene_vec_desc (SceneVecDescription): The scene vector description.
        prompts (list[str]): A list of textual prompts describing scenes.
        scenes (torch.Tensor): The unormalized scenes to evaluate, of shape (B, N, V).
        disable_tqdm (bool): Whether to disable the tqdm progress bar.

    Returns:
        dict[str, float | list[float]]: Dictionary containing the following metrics:
            - num_identifiable_prompts: Number of prompts that could be parsed
            - identifiable_prompt_fraction: Fraction of total prompts that could be
                parsed
            - prompt_following_fraction: Overall fraction of correctly followed prompts
            - object_number_prompt_following_fraction: Fraction for object number
                prompts
            - object_name_prompt_following_fraction: Fraction for object name prompts
            - per_prompt_following_fractions: List of per-prompt following fractions (B,)
            - binary_prompt_satisfaction_rate: Overall fraction of prompts that are
                exactly satisfied
            - binary_object_number_satisfaction_rate: Fraction of object number prompts
                exactly satisfied
            - binary_object_name_satisfaction_rate: Fraction of object name prompts
                exactly satisfied
            - per_prompt_binary_satisfaction: List of binary satisfaction values per
                prompt (B,)
    """
    # Validate inputs.
    if len(prompts) != len(scenes):
        raise ValueError(
            f"Number of prompts and scenes must be the same. "
            f"Got {len(prompts)} prompts and {len(scenes)} scenes."
        )

    num_identifiable_prompts = 0
    num_object_number_prompts = 0
    num_object_name_prompts = 0
    object_number_correct = 0.0
    object_name_correct = 0.0

    # Binary satisfaction metrics.
    binary_object_number_satisfied = 0
    binary_object_name_satisfied = 0

    # List to store per-prompt metrics of shape (B,).
    per_prompt_following_fractions: list[float] = []
    per_prompt_binary_satisfaction: list[int] = []

    for prompt, scene in tqdm(
        zip(prompts, scenes),
        desc="Computing prompt following fraction",
        disable=disable_tqdm,
    ):
        # First check if it's an object number prompt.
        is_obj_num_prompt, gt_obj_num = check_object_number_format(prompt)
        if is_obj_num_prompt:
            num_identifiable_prompts += 1
            num_object_number_prompts += 1

            score, is_satisfied = _evaluate_object_number_prompt(
                scene_vec_desc, scene, gt_obj_num
            )

            object_number_correct += score
            per_prompt_following_fractions.append(score)

            binary_object_number_satisfied += int(is_satisfied)
            per_prompt_binary_satisfaction.append(int(is_satisfied))
            continue

        # Then check if it's an object name prompt.
        is_obj_name_prompt, is_subset = check_object_name_format(prompt)
        if not is_obj_name_prompt:
            # Also include spatial relationship prompts in the object name eval,
            # only evaluating the object name part.
            is_obj_name_prompt, _, _ = check_spatial_relation_format(prompt)
            is_subset = False

        if is_obj_name_prompt:
            num_identifiable_prompts += 1
            num_object_name_prompts += 1

            prompt_score, exact_match = _evaluate_object_name_prompt(
                scene_vec_desc, prompt, scene, is_subset
            )

            object_name_correct += prompt_score
            per_prompt_following_fractions.append(prompt_score)

            binary_object_name_satisfied += int(exact_match)
            per_prompt_binary_satisfaction.append(int(exact_match))
        else:
            # If prompt wasn't identifiable, give full score.
            per_prompt_following_fractions.append(1.0)
            per_prompt_binary_satisfaction.append(1)

    # Calculate final metrics.
    metrics = {
        "num_identifiable_prompts": num_identifiable_prompts,
        "identifiable_prompt_fraction": num_identifiable_prompts / len(prompts),
        "prompt_following_fraction": (
            (object_number_correct + object_name_correct) / num_identifiable_prompts
            if num_identifiable_prompts > 0
            else 0.0
        ),
        "object_number_prompt_following_fraction": (
            object_number_correct / num_object_number_prompts
            if num_object_number_prompts > 0
            else 0.0
        ),
        "object_name_prompt_following_fraction": (
            object_name_correct / num_object_name_prompts
            if num_object_name_prompts > 0
            else 0.0
        ),
        "per_prompt_following_fractions": per_prompt_following_fractions,
        # Binary satisfaction metrics.
        "binary_prompt_satisfaction_rate": (
            (binary_object_number_satisfied + binary_object_name_satisfied)
            / num_identifiable_prompts
            if num_identifiable_prompts > 0
            else 0.0
        ),
        "binary_object_number_satisfaction_rate": (
            binary_object_number_satisfied / num_object_number_prompts
            if num_object_number_prompts > 0
            else 0.0
        ),
        "binary_object_name_satisfaction_rate": (
            binary_object_name_satisfied / num_object_name_prompts
            if num_object_name_prompts > 0
            else 0.0
        ),
        "per_prompt_binary_satisfaction": per_prompt_binary_satisfaction,
    }
    return metrics


def generate_prompt_following_table(
    scene_vec_desc: SceneVecDescription,
    prompts: list[str],
    scenes: torch.Tensor,
    vis_cfg: DictConfig,
    num_workers: int = 1,
) -> wandb.Table:
    """
    Generate a wandb table with columns "Prompt", "Scene Render", "Prompt Type", "Score"
    and number of rows equal to the number of prompts/ scenes.

    Args:
        scene_vec_desc (SceneVecDescription): The scene vector description.
        prompts (list[str]): A list of textual prompts describing scenes of shape (B,).
        scenes (torch.Tensor): The scenes of shape (B, N, V).
        vis_cfg (DictConfig): The visualization configuration with keys
            "camera_pose", "image_width", "image_height", "background_color".
        num_workers (int): The number of workers to use for rendering.

    Returns:
        wandb.Table: A table with the specified columns and rows.
    """
    if len(prompts) != len(scenes):
        raise ValueError("Number of prompts and scenes must be the same.")

    # Render scenes in parallel.
    scene_images = get_scene_renders(
        scenes=scenes,
        scene_vec_desc=scene_vec_desc,
        camera_poses=vis_cfg.camera_pose,
        camera_width=vis_cfg.image_width,
        camera_height=vis_cfg.image_height,
        background_color=vis_cfg.background_color,
        num_workers=num_workers,
        use_blender_server=vis_cfg.use_blender_server,
        blender_server_url=vis_cfg.blender_server_url,
    )

    # Create table.
    table = wandb.Table(columns=["Prompt", "Scene Render", "Prompt Type", "Score"])
    for prompt, scene, scene_image in zip(prompts, scenes, scene_images):
        # Compute metrics for this scene.
        metrics = compute_prompt_following_metrics(
            scene_vec_desc, [prompt], scene.unsqueeze(0)
        )

        # Check the prompt type.
        object_number_prompt_following_fraction = metrics[
            "object_number_prompt_following_fraction"
        ]
        object_name_prompt_following_fraction = metrics[
            "object_name_prompt_following_fraction"
        ]
        score = max(
            object_number_prompt_following_fraction,
            object_name_prompt_following_fraction,
        )
        if object_number_prompt_following_fraction > 0.0:
            prompt_type = "Object Number"
        elif object_name_prompt_following_fraction > 0.0:
            prompt_type = "Object Name"
        else:
            prompt_type = "Unknown"

        # Add row to table.
        table.add_data(prompt, wandb.Image(scene_image), prompt_type, score)

    return table
