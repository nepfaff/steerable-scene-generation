"""
Script for rearranging or completing scenes from a trained model.

Usage:
- Specify `load=...` to load the run ID or local path to the checkpoint. This is
    required.
- Specify `+completion=True` to perform scene completion instead of rearrangement
    (default: False).
- Specify `+num_scenes=...` to set the number of scenes to process (default: 50).
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
    if not is_rank_zero:
        raise ValueError(
            "This script must be run on the main process. "
            "Try export CUDA_VISIBLE_DEVICES=0."
        )

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
    do_completion = cfg.get("completion", False)
    num_scenes = cfg.get("num_scenes", 50)

    # Set predict mode.
    cfg.algorithm.predict.do_sample = False
    cfg.algorithm.predict.do_inference_time_search = False
    cfg.algorithm.predict.do_sample_scenes_with_k_closest_training_examples = False
    cfg.algorithm.predict.do_rearrange = not do_completion
    cfg.algorithm.predict.do_complete = do_completion

    # Ensure that post-processing doesn't reduce the number of scenes.
    cfg.algorithm.postprocessing.return_original_scenes_on_failure = True
    cfg.algorithm.postprocessing.non_penetration_projection.discard_failed_projection_scenes = (
        False
    )

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
    if cfg.wandb.project is None:
        cfg.wandb.project = str(Path(__file__).parent.parent.name)
    name = (
        f"{'completion' if do_completion else 'rearrangement'}_visualization "
        f"({output_dir.parent.name}/{output_dir.name})"
    )
    wandb.init(
        name=name,
        dir=str(output_dir),
        config=OmegaConf.to_container(cfg),
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
    )

    # Load the checkpoint.
    load_id = cfg.load
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
        batch_size=algo.cfg.validation.sample_batch_size or num_scenes,
        num_workers=16,
        shuffle=False,
        persistent_workers=True,
        pin_memory=cfg.experiment.test.pin_memory,
    )

    # Process scenes (either rearrange or complete).
    # Need to rebuild the experiment for this to work.
    experiment = build_experiment(cfg, ckpt_path=checkpoint_path)
    processed_scene_batches = experiment.exec_task("predict", dataloader=dataloader)
    processed_scenes = torch.cat(processed_scene_batches, dim=0)

    # Render scenes.
    logging.info("Rendering original scenes")
    original_renders = get_scene_renders(
        scenes=selected_scenes,
        scene_vec_desc=algo.scene_vec_desc,
        camera_poses=algo.cfg.visualization.camera_pose,
        camera_width=algo.cfg.visualization.image_width,
        camera_height=algo.cfg.visualization.image_height,
        background_color=algo.cfg.visualization.background_color,
        num_workers=algo.cfg.visualization.num_workers,
        use_blender_server=algo.cfg.visualization.use_blender_server,
        blender_server_url=algo.cfg.visualization.blender_server_url,
    )
    logging.info("Rendering processed scenes")
    processed_renders = get_scene_renders(
        scenes=processed_scenes,
        scene_vec_desc=algo.scene_vec_desc,
        camera_poses=algo.cfg.visualization.camera_pose,
        camera_width=algo.cfg.visualization.image_width,
        camera_height=algo.cfg.visualization.image_height,
        background_color=algo.cfg.visualization.background_color,
        num_workers=algo.cfg.visualization.num_workers,
        use_blender_server=algo.cfg.visualization.use_blender_server,
        blender_server_url=algo.cfg.visualization.blender_server_url,
    )

    # Log the results to wandb.
    table = wandb.Table(columns=["Original Scene", "Processed Scene"])
    for i in range(len(processed_renders)):
        table.add_data(
            wandb.Image(original_renders[i]), wandb.Image(processed_renders[i])
        )
    wandb.log({f"{'Completion' if do_completion else 'Rearrangement'} Results": table})

    logging.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
