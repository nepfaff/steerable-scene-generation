"""
Script for running full evaluation on one or more checkpoints from a wandb run.

NOTE:
- Specify `load=...` to load the run ID or local path to the checkpoint. This is
    required.
- Specify `+output_dir=...` to resume previous eval/ re-use the already rendered
    scenes.
- Specify `+conditional=True` for conditional sampling based on the labels in the GT
    samples.
- Specify `checkpoint_version=...` to download a specific version or a list of versions.
- Specify `+prompt_following_table_length=...` to specify the number of samples to
    include in the prompt following table. Only used if `+conditional=True`. Defaults
    to no table being logged. Note that this might result into buffer overflow errors
    when multiple checkpoints are evaluated.
- Specify `+include_image_metrics=False` to exclude image metrics in the evaluation.
    This is faster as we don't need to render any images.
- Specify `+num_renders=...` to specify the number of renders to log. Defaults to None.
    Note that this might result into buffer overflow errors when multiple checkpoints
    are evaluated.
- Specify `+num_samples=...` to specify the number of samples to evaluate. Defaults to
    5000.
- Specify `+batch_size=...` to specify the sample batch size for evaluation. Defaults to
    256.
- Specify `+render_batch_size=...` to specify the render batch size for evaluation.
    Defaults to 500.
- Specify `+num_image_ca_repeats=...` to specify the number of times to repeat the
    image CA metric. Defaults to 10.
- Specify `+num_scene_ca_repeats=...` to specify the number of times to repeat the
    scene CA metric. Defaults to 0.

Also note that there might be issues (e.g. getting stuck) when the number of samples is
not divisible by the number of GPU workers.
"""

import io
import logging
import os
import pickle

from pathlib import Path
from typing import Dict, List

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from cleanfid import fid
from cmmd_pytorch import compute_cmmd
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict
from PIL import Image
from prdc import compute_prdc
from top_pr import compute_top_pr
from torch import nn
from tqdm import tqdm

from steerable_scene_generation.algorithms.common.dataclasses import SceneVecDescription
from steerable_scene_generation.algorithms.scene_diffusion.scene_diffuser_base import (
    SceneDiffuserBase,
)
from steerable_scene_generation.experiments import build_experiment
from steerable_scene_generation.utils.ckpt_utils import (
    download_all_checkpoints,
    download_version_checkpoint,
    is_run_id,
)
from steerable_scene_generation.utils.distributed_utils import is_rank_zero
from steerable_scene_generation.utils.image_ca_metric import compute_image_ca_metric
from steerable_scene_generation.utils.logging import (
    filter_drake_obj_warning,
    filter_drake_vtk_warning,
)
from steerable_scene_generation.utils.omegaconf import register_resolvers
from steerable_scene_generation.utils.prompt_following_metrics import (
    compute_prompt_following_metrics,
    generate_prompt_following_table,
)
from steerable_scene_generation.utils.scene_metrics import (
    compute_scene_object_kl_divergence_metric,
    compute_welded_object_pose_deviation_metric,
)
from steerable_scene_generation.utils.scene_vector_ca_metric import (
    compute_scene_vector_ca_metric,
)
from steerable_scene_generation.utils.visualization import (
    get_scene_label_image_renders,
    get_scene_renders,
    setup_virtual_display_if_needed,
)

# Add logging filters.
filter_drake_vtk_warning()
filter_drake_obj_warning()

# Disable tokenizer parallelism.
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RandomSceneFeatureExtractor(nn.Module):
    """
    Meant to be randomly initialized for extracting features from scene vectors.
    Inspired by https://arxiv.org/abs/2002.09797.
    """

    def __init__(self, input_dim: int, hidden_dim: int, feature_dim: int):
        super().__init__()

        # Phi network applied to each object independently.
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Rho network applied after object aggregation.
        self.rho = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, V = x.size()

        # Apply phi to each object independently.
        x = x.view(B * N, V)  # Shape (B * N, V)
        x = self.phi(x)
        x = x.view(B, N, -1)  # Shape (B, N, hidden_dim)

        # Mean and max pooling (permutation-invariant).
        x_mean = x.mean(dim=1)  # Shape (B, hidden_dim)
        x_max = x.max(dim=1).values  # Shape (B, hidden_dim)
        x = torch.cat([x_mean, x_max], dim=-1)  # Shape (B, 2*hidden_dim)

        # Apply rho to the aggregated representation.
        x = self.rho(x)  # Shape (B, feature_dim)

        # Normalize the features.
        x = torch.nn.functional.normalize(x, dim=-1)  # Shape (B, feature_dim)

        return x


def render_and_save_images(
    algo_cfg, render_dir: str, scene_vec_desc: SceneVecDescription, scenes: torch.Tensor
) -> None:
    os.makedirs(render_dir, exist_ok=True)

    # Render and save in batches.
    counter = 0
    render_batch_size = algo_cfg.get("render_batch_size", 500)
    scene_batches = np.array_split(scenes, np.ceil(len(scenes) / render_batch_size))
    for scenes in tqdm(scene_batches, desc="Rendering scene batches.", leave=False):
        dataset_images_array = get_scene_label_image_renders(
            scenes=scenes,
            scene_vec_desc=scene_vec_desc,
            camera_poses=algo_cfg.visualization.camera_pose,
            camera_width=algo_cfg.visualization.image_width,
            camera_height=algo_cfg.visualization.image_height,
            num_workers=algo_cfg.visualization.num_workers,
        )

        # Save images to disk.
        for img_np in tqdm(
            dataset_images_array, desc="Saving scene render images", leave=False
        ):
            # Convert to uint8.
            img_normalized = (img_np * 255).clip(0, 255)
            img_uint8 = img_normalized.astype(np.uint8)

            img = Image.fromarray(img_uint8)
            path = os.path.join(render_dir, f"semantic_render_{counter:04}.png")
            img.save(path)

            counter += 1


def log_images(
    image_dir: str, num_images_to_log: int, log_path: str, global_step: int
) -> None:
    """Logs the first `num_images_to_log` images in the given directory to wandb."""
    # Sort and pick first N files.
    image_files = sorted(os.listdir(image_dir))[:num_images_to_log]

    # Create the wandb images.
    images = []
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        if image_file.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            image = Image.open(image_path)
            images.append(wandb.Image(image))

    # Log to wandb.
    wandb.log({log_path: images}, step=global_step)


def render_and_log_images(
    algo_cfg,
    scene_vec_desc: SceneVecDescription,
    scenes: torch.Tensor,
    camera_poses: dict[str, dict[str, list[float]]],
    global_step: int,
) -> None:
    """Renders and logs images of the scenes to wandb."""
    # Render.
    images_array = get_scene_renders(
        scenes=scenes,
        scene_vec_desc=scene_vec_desc,
        camera_poses=camera_poses,
        camera_width=algo_cfg.visualization.image_width,
        camera_height=algo_cfg.visualization.image_height,
        num_workers=algo_cfg.visualization.num_workers,
        use_blender_server=algo_cfg.visualization.use_blender_server,
        blender_server_url=algo_cfg.visualization.blender_server_url,
    )  # Shape (B, H, W, 3)

    # Log to wandb.
    images = []
    for image in images_array:
        images.append(wandb.Image(image))
    wandb.log({"semantic/samples": images}, step=global_step)


@hydra.main(version_base=None, config_path="../configurations", config_name="config")
def main(cfg: DictConfig):
    setup_virtual_display_if_needed()

    # Resolve the config.
    register_resolvers()
    OmegaConf.resolve(cfg)

    # Set up unconditional or conditional sampling.
    is_conditional = cfg.get("conditional", False)
    if is_conditional:
        if not cfg.algorithm.classifier_free_guidance.use:
            raise ValueError(
                "Conditional sampling requested but classifier free guidance is not "
                "enabled."
            )
        cfg.algorithm.classifier_free_guidance.sampling.use_data_labels = True
        cfg.dataset.classifier_free_guidance.sampling.use_data_labels = True
        logging.info("Running conditional sampling with the dataset labels.")
    else:
        cfg.algorithm.classifier_free_guidance.sampling.use_data_labels = False
        cfg.algorithm.classifier_free_guidance.sampling.labels = ""
        cfg.algorithm.classifier_free_guidance.weight = -1.0
        cfg.dataset.classifier_free_guidance.sampling.use_data_labels = False
        cfg.dataset.classifier_free_guidance.sampling.labels = ""
        logging.info("Running unconditional sampling.")
    # Set masking probs to zero.
    cfg.algorithm.classifier_free_guidance.masking_prob = 0.0
    cfg.dataset.classifier_free_guidance.masking_prob = 0.0
    cfg.algorithm.classifier_free_guidance.masking_prob_coarse = 0.0
    cfg.dataset.classifier_free_guidance.masking_prob_coarse = 0.0
    # Set predict mode.
    cfg.algorithm.predict.do_sample = True
    cfg.algorithm.predict.do_rearrange = False
    cfg.algorithm.predict.do_complete = False
    cfg.algorithm.predict.do_inference_time_search = False
    cfg.algorithm.predict.do_sample_scenes_with_k_closest_training_examples = False

    # Get yaml names.
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    cfg_choice = OmegaConf.to_container(hydra_cfg.runtime.choices)

    with open_dict(cfg):
        if cfg_choice["experiment"] is not None:
            cfg.experiment._name = cfg_choice["experiment"]
        if cfg_choice["dataset"] is not None:
            cfg.dataset._name = cfg_choice["dataset"]
        if cfg_choice["algorithm"] is not None:
            cfg.algorithm._name = cfg_choice["algorithm"]

    # Set up the output directory.
    output_dir = Path(hydra_cfg.runtime.output_dir)
    if is_rank_zero:
        logging.info(f"Outputs will be saved to: {output_dir}")
        (output_dir.parents[1] / "latest-run").unlink(missing_ok=True)
        (output_dir.parents[1] / "latest-run").symlink_to(
            output_dir, target_is_directory=True
        )

    load_id = cfg.get("load", None)
    assert load_id is not None, "Must provide a load ID."

    if cfg.wandb.project is None:
        cfg.wandb.project = str(Path(__file__).parent.parent.name)

    # Set up logging with wandb.
    name = f"eval_{load_id} ({output_dir.parent.name}/{output_dir.name})"
    save_dir = cfg.output_dir if "output_dir" in cfg else str(output_dir)
    logging.info(f"Using save_dir {save_dir}")
    if is_rank_zero:
        wandb.init(
            name=name,
            dir=save_dir,
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg),
            mode=cfg.wandb.mode,
        )

    run_path = f"{cfg.wandb.entity}/{cfg.wandb.project}/{load_id}"
    download_dir = Path(save_dir) / "checkpoints"
    if is_run_id(load_id):  # Download checkpoints from wandb.
        version = cfg.checkpoint_version
        if version is not None and isinstance(version, int):
            checkpoint_path = download_version_checkpoint(
                run_path=run_path, version=version, download_dir=download_dir
            )
            checkpoint_infos = [{"path": checkpoint_path, "version": version}]
        else:
            checkpoint_infos = download_all_checkpoints(
                run_path=run_path, download_dir=download_dir, versions=version
            )
        delete_checkpoints_after_eval = True
    else:  # Load from local path.
        checkpoint_path = Path(load_id)
        checkpoint_infos = [{"path": checkpoint_path, "version": None}]
        delete_checkpoints_after_eval = False

    if is_rank_zero:
        # Add global_step info.
        for ckpt_info in checkpoint_infos:
            ckpt = torch.load(ckpt_info["path"])
            ckpt_info["global_step"] = ckpt["global_step"]

    # Set random seed.
    seed = cfg.experiment.seed if cfg.experiment.seed is not None else 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    logging.info(f"Using seed {seed} for evaluation.")

    # Select dataset scenes.
    num_samples = cfg.get("num_samples", 5000)
    first_ckpt_path = checkpoint_infos[0]["path"]
    experiment = build_experiment(cfg=cfg, ckpt_path=first_ckpt_path)
    algo: SceneDiffuserBase = experiment._build_algo(ckpt_path=first_ckpt_path)
    # Determine the total number of scenes and sample indices.
    if algo.dataset.use_subdataset_sampling:
        logging.info("Using subdataset sampling for the GT scenes.")
        # Add dataset indices that respect the subdataset proportions.
        indices = []
        for subdataset_idx in range(len(algo.dataset.subdataset_ranges)):
            # Get the probability for this subdataset.
            name = algo.dataset.subdataset_names[subdataset_idx]
            prob = algo.dataset.subdataset_probs[name]
            # Calculate number of samples based on total desired samples, rounded down.
            num_samples = int(num_samples * prob)
            start_idx, end_idx = algo.dataset.subdataset_ranges[subdataset_idx]
            new_samples = torch.from_numpy(
                np.random.choice(range(start_idx, end_idx), num_samples, replace=False)
            )
            indices.append(new_samples)
        indices = torch.cat(indices)
        if len(indices) < num_samples:
            # Add random indices to get the correct number of samples.
            num_missing = num_samples - len(indices)
            added_indices = torch.from_numpy(
                np.random.choice(len(algo.dataset), size=num_missing, replace=False)
            )
            indices = torch.cat([indices, added_indices])
            logging.info(
                f"Added {num_missing} random indices to get {len(indices)} samples."
            )
    else:
        logging.info("Using random sampling for the GT scenes.")
        # Random indices.
        indices = torch.from_numpy(
            np.random.choice(len(algo.dataset), size=num_samples, replace=False)
        )
    # Fetch data based on the dataset structure.
    selected_data = algo.dataset.get_all_data(normalized=False, scene_indices=indices)
    selected_scenes = selected_data["scenes"]
    assert len(selected_scenes) >= num_samples
    if is_conditional and "language_annotation" not in selected_data:
        raise ValueError(
            "Conditional sampling requested but dataset does not have language "
            "annotations."
        )

    include_image_metrics = cfg.get("include_image_metrics", True)
    if include_image_metrics:
        dataset_semantic_dir = os.path.join(save_dir, "semantic/dataset")

        if is_rank_zero:
            if os.path.exists(dataset_semantic_dir):
                logging.info(
                    f"{dataset_semantic_dir} already exists, skipping rendering"
                )
            else:
                # Render dataset scenes.
                render_and_save_images(
                    algo_cfg=algo.cfg,
                    render_dir=dataset_semantic_dir,
                    scene_vec_desc=algo.scene_vec_desc,
                    scenes=selected_scenes,
                )

            log_images(
                dataset_semantic_dir,
                num_images_to_log=50,
                log_path="semantic/dataset",
                global_step=0,
            )
    else:
        logging.warning("Skipping image metrics due to `include_image_metrics=False`.")

    # Add scene indices to the selected data. These are used for preservering prediction
    # order during gathering across GPUs.
    selected_data["scene_indices"] = torch.arange(len(selected_scenes))

    # Build dataloader for sampling. Use the same scenes as for the dataset renders.
    dataset = algo.dataset
    dataset.set_data(data=selected_data, normalized=False)
    sample_batch_size = cfg.get("batch_size", 256)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=sample_batch_size,
        num_workers=16,
        shuffle=False,
        persistent_workers=True,
        pin_memory=cfg.experiment.test.pin_memory,
    )

    # Sampling checkpoint scenes.
    checkpoint_samples: List[torch.Tensor] = []
    ran_parallel = False
    for checkpoint_info in tqdm(
        checkpoint_infos, desc="Sampling and rendering checkpoints", leave=False
    ):
        version = checkpoint_info["version"]
        if is_rank_zero:
            logging.info(f"Sampling and rendering checkpoint {version}.")

        render_dir = os.path.join(save_dir, "semantic/samples", str(version))
        sample_pkl_path = render_dir + ".pkl"
        if is_rank_zero and os.path.exists(sample_pkl_path):
            logging.info(
                f"Pickle path {sample_pkl_path} already exists. Skipping sampling "
                f"for checkpoint v{version}"
            )
            # Load samples.
            with open(sample_pkl_path, "rb") as f:
                samples = torch.from_numpy(pickle.load(f))
                logging.info(
                    f"Loaded samples for checkopoint v{version} from {sample_pkl_path}"
                )
        else:
            experiment = build_experiment(cfg=cfg, ckpt_path=checkpoint_info["path"])
            experiment.exec_task("predict", dataloader=dataloader)
            ran_parallel = True
            algo: SceneDiffuserBase = experiment.algo

            if is_rank_zero:
                samples = algo.predictions
                if len(samples) != num_samples:
                    logging.warning(
                        f"Got {len(samples)} samples but expected {num_samples}!"
                    )

                # Save samples to disk.
                logging.info(
                    f"Saving version {version} sample pickle. This may take a while..."
                )
                os.makedirs(os.path.dirname(sample_pkl_path), exist_ok=True)
                with open(sample_pkl_path, "wb") as f:
                    pickle.dump(samples.cpu().numpy(), f)
                wandb.save(sample_pkl_path)
                logging.info(f"Saved version {version} samples to {sample_pkl_path}")

        if is_rank_zero:
            checkpoint_samples.append(samples)

        if include_image_metrics and is_rank_zero:
            if os.path.exists(render_dir):
                logging.info(f"{render_dir} already exists, skipping rendering")
            else:
                # Render.
                render_and_save_images(
                    algo_cfg=algo.cfg,
                    render_dir=render_dir,
                    scene_vec_desc=algo.scene_vec_desc,
                    scenes=samples,
                )

        if ran_parallel:
            algo.trainer.strategy.barrier()

    if is_rank_zero:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        scene_feature_extractor = RandomSceneFeatureExtractor(
            input_dim=selected_scenes.shape[-1], hidden_dim=128, feature_dim=64
        ).to(device)

        # Compute checkpoint metrics.
        metrics: List[Dict[str, float]] = []
        for checkpoint_info, samples in tqdm(
            zip(checkpoint_infos, checkpoint_samples),
            desc="Evaluating checkpoints",
            leave=False,
        ):
            version = checkpoint_info["version"]
            global_step = checkpoint_info["global_step"]
            logging.info(
                f"Running full evaluation on checkpoint {version} (step={global_step})."
            )

            if include_image_metrics:
                sample_semantic_dir = os.path.join(
                    save_dir, "semantic/samples", str(version)
                )

                # Compute FID, KID, CMMD metrics.
                fid_score = fid.compute_fid(
                    dataset_semantic_dir,
                    sample_semantic_dir,
                    device=device,
                    verbose=False,
                )
                kid_score = fid.compute_kid(
                    dataset_semantic_dir,
                    sample_semantic_dir,
                    device=device,
                    verbose=False,
                )
                cmmd_score = compute_cmmd(
                    dataset_semantic_dir, sample_semantic_dir
                ).item()

            # Compute various precision and recall metrics on the scene vectors.
            # Using random feature extractor.
            with torch.no_grad():
                dataset_features = (
                    scene_feature_extractor(selected_scenes.to(device)).cpu().numpy()
                )
                sample_features = (
                    scene_feature_extractor(samples.to(device)).cpu().numpy()
                )
            prdc_metrics = compute_prdc(
                real_features=dataset_features,
                fake_features=sample_features,
                nearest_k=5,
            )
            try:
                toppr_metrics = compute_top_pr(
                    real_features=dataset_features, fake_features=sample_features
                )
            except Exception as e:
                # Can fail if the number of samples is too small.
                if is_rank_zero:
                    logging.warning(f"Error computing TopPR: {e}")
                toppr_metrics = {
                    "fidelity": torch.nan,
                    "diversity": torch.nan,
                    "Top_F1": torch.nan,
                }
            # Using simple sum across object dimension aggregation.
            prdc_metrics_simple = compute_prdc(
                real_features=selected_scenes.sum(dim=1).cpu().numpy(),
                fake_features=samples.sum(dim=1).cpu().numpy(),
                nearest_k=5,
            )
            try:
                toppr_metrics_simple = compute_top_pr(
                    real_features=selected_scenes.sum(dim=1).cpu().numpy(),
                    fake_features=samples.sum(dim=1).cpu().numpy(),
                )
            except Exception as e:
                # Can fail if the number of samples is too small.
                if is_rank_zero:
                    logging.warning(f"Error computing TopPR simple: {e}")
                toppr_metrics_simple = {
                    "fidelity": torch.nan,
                    "diversity": torch.nan,
                    "Top_F1": torch.nan,
                }

            # Compute object KL divergence metric.
            kl_divergence = compute_scene_object_kl_divergence_metric(
                dataset_scenes=selected_scenes,
                dataset_scene_vec_desc=algo.scene_vec_desc,
                synthesized_scenes=samples.to(selected_scenes.device),
                synthesized_scene_vec_desc=algo.scene_vec_desc,
            )

            # Compute welded object pose deviation metric.
            welded_object_pose_deviation = compute_welded_object_pose_deviation_metric(
                dataset_scenes=selected_scenes,
                dataset_scene_vec_desc=algo.scene_vec_desc,
                synthesized_scenes=samples.to(selected_scenes.device),
                synthesized_scene_vec_desc=algo.scene_vec_desc,
            )

            num_image_ca_repeats = cfg.get("num_image_ca_repeats", 10)
            if include_image_metrics and num_image_ca_repeats > 0:
                # Compute semantic image CA metric.
                image_ca_iterations = [25, 50, 100, 150, 200]
                image_ca_results = compute_image_ca_metric(
                    dataset_directory=dataset_semantic_dir,
                    synthesized_directory=sample_semantic_dir,
                    num_runs=num_image_ca_repeats,
                    iterations=image_ca_iterations,
                )
            else:
                image_ca_iterations = []
                image_ca_results = []

            num_scene_ca_repeats = cfg.get("num_scene_ca_repeats", 0)
            if num_scene_ca_repeats > 0:
                # Compute scene vector CA metric.
                scene_vector_ca_epochs = [10, 20, 30, 40, 50, 75, 100, 150]
                scene_vector_ca_results = compute_scene_vector_ca_metric(
                    dataset_scenes=selected_scenes,
                    synthesized_scenes=samples,
                    epochs=scene_vector_ca_epochs,
                    num_runs=num_scene_ca_repeats,
                    use_transformer=True,
                )
            else:
                scene_vector_ca_epochs = []
                scene_vector_ca_results = []

            # Construct metric dict.
            checkpoint_metrics = {
                "global_step": global_step,
                # KL divergence.
                "validation/object_kl_divergence": kl_divergence,
                # Welded object pose deviation.
                "validation/welded_object_pose_deviation": welded_object_pose_deviation,
                # PRDC metrics.
                "validation/prdc_precision": prdc_metrics["precision"],
                "validation/prdc_recall": prdc_metrics["recall"],
                "validation/prdc_density": prdc_metrics["density"],
                "validation/prdc_coverage": prdc_metrics["coverage"],
                # TOPPR metrics.
                "validation/toppr_fidelity": toppr_metrics["fidelity"],
                "validation/toppr_diversity": toppr_metrics["diversity"],
                "validation/toppr_f1": toppr_metrics["Top_F1"],
                # PRDC 'simple' metrics.
                "validation/prdc_simple_precision": prdc_metrics_simple["precision"],
                "validation/prdc_simple_recall": prdc_metrics_simple["recall"],
                "validation/prdc_simple_density": prdc_metrics_simple["density"],
                "validation/prdc_simple_coverage": prdc_metrics_simple["coverage"],
                # TOPPR 'simple' metrics.
                "validation/toppr_simple_fidelity": toppr_metrics_simple["fidelity"],
                "validation/toppr_simple_diversity": toppr_metrics_simple["diversity"],
                "validation/toppr_simple_f1": toppr_metrics_simple["Top_F1"],
            }
            if include_image_metrics:
                # Add image metrics.
                checkpoint_metrics.update(
                    {
                        "validation/fid": fid_score,
                        "validation/kid": kid_score,
                        "validation/cmmd": cmmd_score,
                    }
                )
                # Add image CA metrics.
                for iteration, res in zip(image_ca_iterations, image_ca_results):
                    checkpoint_metrics[
                        f"validation/image_ca_mean_{iteration}_iterations"
                    ] = res[0]
                    checkpoint_metrics[
                        f"validation/image_ca_std_{iteration}_iterations"
                    ] = res[1]
            # Scene vector CA metrics.
            for epoch, res in zip(scene_vector_ca_epochs, scene_vector_ca_results):
                checkpoint_metrics[f"validation/vector_ca_mean_{epoch}_epochs"] = res[0]
                checkpoint_metrics[f"validation/vector_ca_std_{epoch}_epochs"] = res[1]

            # Compute physical feasibility metrics.
            feasibility_metrics = algo.compute_physical_feasibility_metrics(
                scenes=samples, num_scenes=len(samples), name="validation"
            )
            checkpoint_metrics.update(feasibility_metrics)

            # Compute the prompt following metrics.
            if is_conditional:
                prompts = selected_data["language_annotation"]
                prompt_following_metrics = compute_prompt_following_metrics(
                    scene_vec_desc=algo.scene_vec_desc, prompts=prompts, scenes=samples
                )
                for metric_name, metric_value in prompt_following_metrics.items():
                    checkpoint_metrics[f"validation/{metric_name}"] = metric_value

                prompt_following_table_length = cfg.get(
                    "prompt_following_table_length", None
                )
                if prompt_following_table_length is not None:
                    prompt_following_table = generate_prompt_following_table(
                        scene_vec_desc=algo.scene_vec_desc,
                        prompts=prompts[:prompt_following_table_length],
                        scenes=samples[:prompt_following_table_length],
                        vis_cfg=cfg.algorithm.visualization,
                        num_workers=cfg.algorithm.visualization.num_workers,
                    )
                    wandb.log(
                        {
                            f"prompt_following_table_{global_step:07}": prompt_following_table,
                        },
                        step=global_step,
                    )
                else:
                    logging.info("Skipping prompt following table logging.")

            metrics.append(checkpoint_metrics)
            wandb.log(checkpoint_metrics, step=global_step)

            # Log sample images to wandb.
            if include_image_metrics:
                log_images(
                    sample_semantic_dir,
                    num_images_to_log=50,
                    log_path=os.path.join("semantic/samples", str(global_step)),
                    global_step=global_step,
                )
            num_renders = cfg.get("num_renders", None)
            if num_renders is not None:
                render_and_log_images(
                    algo_cfg=algo.cfg,
                    scene_vec_desc=algo.scene_vec_desc,
                    scenes=samples[:num_renders],
                    camera_poses=algo.cfg.visualization.camera_pose,
                    global_step=global_step,
                )

        # Log the metrics in table form.
        wandb.log(
            {
                "metrics": wandb.Table(
                    columns=list(metrics[0].keys()),
                    data=[list(metric.values()) for metric in metrics],
                )
            }
        )

        if len(metrics) > 1:
            # Plot metrics.
            metric_names = [key for key in metrics[0].keys() if key != "global_step"]
            global_steps = [float(d["global_step"]) for d in metrics]

            for metric_name in metric_names:
                if "_std" in metric_name:
                    continue

                # Check if there's an associated std for the metric to create an error
                # bar plot.
                if (
                    "_mean" in metric_name
                    and metric_name.replace("_mean", "_std") in metric_names
                ):
                    std_metric_name = metric_name.replace("_mean", "_std")
                    values = [float(d[metric_name]) for d in metrics]
                    std_values = [float(d[std_metric_name]) for d in metrics]

                    # Create a plot with interpolated error bars.
                    fig = plt.figure()
                    plt.plot(global_steps, values, label=metric_name, color="b")
                    plt.fill_between(
                        global_steps,
                        np.array(values) - np.array(std_values),
                        np.array(values) + np.array(std_values),
                        color="b",
                        alpha=0.2,
                    )
                    plt.xlabel("Training Step")
                    plt.ylabel(metric_name.capitalize())
                    plt.title(f"{metric_name.capitalize()}")
                    plt.grid(True)

                    # Save the error bar plot to a BytesIO object.
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)

                    # Log to wandb.
                    img = Image.open(buf)
                    wandb.log(
                        {metric_name.capitalize() + "_error_bar_plot": wandb.Image(img)}
                    )

                    # Close the figure and buffer.
                    plt.close(fig)
                    buf.close()

        if delete_checkpoints_after_eval:
            # Delete all downloaded checkpoints.
            for checkpoint_info in checkpoint_infos:
                checkpoint_info["path"].unlink()
            logging.info("Deleted all downloaded checkpoints.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
