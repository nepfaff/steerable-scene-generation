"""
Script for sampling scenes from a trained model and finding the k closest training
examples in the dataset. The closest examples are determined using the optimal
transport distance.

Usage:
- Specify `load=...` to load the run ID or local path to the checkpoint. This is
    required.
- Specify `algorithm.predict.sample_scenes_with_k_closest_training_examples.num_k=...`
    to set the number of closest training examples to find.
- Specify `algorithm.predict.sample_scenes_with_k_closest_training_examples.batch_size=...`
    to set the batch size computing the optimal transport distance.
- Specify `+num_scenes=...` to set the number of scenes to sample (default: 8).
"""

import logging
import os

from pathlib import Path

import hydra
import numpy as np
import torch
import wandb

from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict

from steerable_scene_generation.algorithms.scene_diffusion.scene_diffuser_base import (
    SceneDiffuserBase,
)
from steerable_scene_generation.experiments import build_experiment
from steerable_scene_generation.utils.ckpt_utils import (
    download_latest_or_best_checkpoint,
    is_run_id,
)
from steerable_scene_generation.utils.distributed_utils import is_rank_zero
from steerable_scene_generation.utils.logging import filter_drake_vtk_warning
from steerable_scene_generation.utils.omegaconf import register_resolvers
from steerable_scene_generation.utils.visualization import (
    get_scene_renders,
    setup_virtual_display_if_needed,
)

# Add logging filters.
filter_drake_vtk_warning()

# Disable tokenizer parallelism.
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@hydra.main(version_base=None, config_path="../configurations", config_name="config")
def main(cfg: DictConfig) -> None:
    if not cfg.algorithm.visualization.use_blender_server:
        setup_virtual_display_if_needed()

    # Set random seed.
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Resolve the config.
    register_resolvers()
    OmegaConf.resolve(cfg)

    # Get configuration values with defaults.
    num_scenes = cfg.get("num_scenes", 8)

    # Set predict mode.
    cfg.algorithm.predict.do_sample = False
    cfg.algorithm.predict.do_rearrange = False
    cfg.algorithm.predict.do_complete = False
    cfg.algorithm.predict.do_inference_time_search = False
    cfg.algorithm.predict.do_sample_scenes_with_k_closest_training_examples = True

    # Check if load path is provided.
    if "load" not in cfg or cfg.load is None:
        raise ValueError("Please specify a checkpoint to load with 'load=...'")

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
    logging.info(f"Outputs will be saved to: {output_dir}")
    (output_dir.parents[1] / "latest-run").unlink(missing_ok=True)
    (output_dir.parents[1] / "latest-run").symlink_to(
        output_dir, target_is_directory=True
    )

    # Initialize wandb.
    load_id = cfg.load
    if cfg.wandb.project is None:
        cfg.wandb.project = str(Path(__file__).parent.parent.name)
    name = (
        f"sample_and_find_k_closest_training_examples_{load_id} "
        f"({output_dir.parent.name}/{output_dir.name})"
    )
    if is_rank_zero:
        wandb.init(
            name=name,
            dir=str(output_dir),
            config=OmegaConf.to_container(cfg),
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
        )

    # Load the checkpoint.
    if is_run_id(load_id):
        # Download the checkpoint from wandb.
        run_path = f"{cfg.wandb.entity}/{cfg.wandb.project}/{load_id}"
        download_dir = output_dir / "checkpoints"
        checkpoint_path = download_latest_or_best_checkpoint(
            run_path=run_path, download_dir=download_dir, use_best=cfg.use_best
        )
    else:
        # Use local path.
        checkpoint_path = Path(load_id)

    # Build the experiment.
    experiment = build_experiment(cfg, ckpt_path=checkpoint_path)

    # Get the algorithm.
    algo: SceneDiffuserBase = experiment._build_algo(ckpt_path=checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algo = algo.to(device)

    # Select the scenes and build the dataloader.
    indices = torch.from_numpy(
        np.random.choice(len(algo.dataset), size=num_scenes, replace=False)
    )
    selected_data = algo.dataset.get_all_data(normalized=False, scene_indices=indices)
    selected_scenes = selected_data["scenes"]
    assert len(selected_scenes) >= num_scenes
    dataset = algo.dataset
    dataset.set_data(data=selected_data, normalized=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=16,
        shuffle=False,
        persistent_workers=True,
        pin_memory=cfg.experiment.test.pin_memory,
    )

    # Sample scenes and find the k closest training examples.
    # Need to rebuild the experiment for this to work.
    experiment = build_experiment(cfg, ckpt_path=checkpoint_path)
    experiment.exec_task("predict", dataloader=dataloader)
    algo: SceneDiffuserBase = experiment.algo

    # Wait for all processes to finish.
    algo.trainer.strategy.barrier()

    if is_rank_zero:
        # Get the results.
        results: dict[str, torch.Tensor] = algo.predictions
        sampled_scenes = results["sampled_scenes"]
        closest_training_scenes = results["closest_training_scenes"]
        distances = results["distances"]

        # Check that everything went correctly.
        assert sampled_scenes.dim() == 3  # (B, N, V)
        assert closest_training_scenes.dim() == 4  # (B, K, N, V)
        assert distances.dim() == 2  # (B, K)

        # Render scenes.
        logging.info("Rendering sampled scenes")
        sampled_renders = get_scene_renders(
            scenes=sampled_scenes,
            scene_vec_desc=algo.scene_vec_desc,
            camera_poses=algo.cfg.visualization.camera_pose,
            camera_width=algo.cfg.visualization.image_width,
            camera_height=algo.cfg.visualization.image_height,
            background_color=algo.cfg.visualization.background_color,
            num_workers=algo.cfg.visualization.num_workers,
            use_blender_server=algo.cfg.visualization.use_blender_server,
            blender_server_url=algo.cfg.visualization.blender_server_url,
        )  # Shape (B, H, W, 4)
        logging.info("Rendering closest training scenes")
        closest_training_scenes_flat = closest_training_scenes.view(
            -1, *closest_training_scenes.shape[2:]
        )  # Shape (B * K, H, W, 4)
        closest_training_renders = get_scene_renders(
            scenes=closest_training_scenes_flat,
            scene_vec_desc=algo.scene_vec_desc,
            camera_poses=algo.cfg.visualization.camera_pose,
            camera_width=algo.cfg.visualization.image_width,
            camera_height=algo.cfg.visualization.image_height,
            background_color=algo.cfg.visualization.background_color,
            num_workers=algo.cfg.visualization.num_workers,
            use_blender_server=algo.cfg.visualization.use_blender_server,
            blender_server_url=algo.cfg.visualization.blender_server_url,
        ).reshape(
            closest_training_scenes.shape[0],
            closest_training_scenes.shape[1],
            algo.cfg.visualization.image_height,
            algo.cfg.visualization.image_width,
            4,
        )  # Shape (B, K, H, W, 4)

        # Log the results as a table to wandb.
        columns = ["Sampled Scene"]
        for i in range(closest_training_scenes.shape[1]):
            columns.append(f"Closest Training Scene {i+1}")
            columns.append(f"Distance {i+1}")
        table = wandb.Table(columns=columns)
        for main_image, associated_images, distance_scores in zip(
            sampled_renders, closest_training_renders, distances
        ):
            row_data = [wandb.Image(main_image)]
            for img, score in zip(associated_images, distance_scores):
                row_data.append(wandb.Image(img))
                row_data.append(score)
            table.add_data(*row_data)
        wandb.log({"Results": table})

        logging.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
