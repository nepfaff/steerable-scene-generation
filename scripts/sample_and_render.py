"""
Script for sampling from a trained model and rendering the sampled scenes.
This enables sampling without loading a dataset.

Usage:
- Specify `load=...` to load the run ID or local path to the checkpoint. This is
    required.
- Specify `+num_scenes=...` to set the number of scenes to sample (default: 50).
- Specify `+save_pickle=True` to save the sampled scenes as a pickle file to wandb
    instead of rendering them.
"""

import logging
import os
import pickle

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
    download_version_checkpoint,
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

    # Check if load path is provided.
    if "load" not in cfg or cfg.load is None:
        raise ValueError("Please specify a checkpoint to load with 'load=...'")

    # Get configuration values with defaults.
    num_scenes = cfg.get("num_scenes", 50)

    # Set predict mode.
    cfg.algorithm.predict.do_sample = True
    cfg.algorithm.predict.do_inference_time_search = False
    cfg.algorithm.predict.do_sample_scenes_with_k_closest_training_examples = False
    cfg.algorithm.predict.do_rearrange = False
    cfg.algorithm.predict.do_complete = False

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
    load_id = cfg.load
    name = (
        f"sampling_visualization_{load_id} ({output_dir.parent.name}/{output_dir.name})"
    )
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
        version = cfg.checkpoint_version
        if version is not None and isinstance(version, int):
            checkpoint_path = download_version_checkpoint(
                run_path=run_path, version=version, download_dir=download_dir
            )
        else:
            checkpoint_path = download_latest_or_best_checkpoint(
                run_path=run_path, download_dir=download_dir, use_best=cfg.use_best
            )
    else:
        # Use local path.
        checkpoint_path = Path(load_id)

    # Get the algorithm.
    experiment = build_experiment(cfg, ckpt_path=checkpoint_path)
    algo: SceneDiffuserBase = experiment._build_algo(ckpt_path=checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algo = algo.to(device)

    # Build a fake dataloader. We need one.
    num_objects_per_scene = (
        algo.cfg.max_num_objects_per_scene + algo.cfg.num_additional_tokens_for_sampling
    )
    fake_data = {
        "scenes": torch.randn(
            num_scenes, num_objects_per_scene, algo.scene_vec_desc.get_object_vec_len()
        ),
        "language_annotation": [""] * num_scenes,
    }
    dataset = algo.dataset
    dataset.set_data(data=fake_data, normalized=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=algo.cfg.validation.sample_batch_size or num_scenes,
        num_workers=16,
        shuffle=False,
        persistent_workers=True,
        pin_memory=cfg.experiment.test.pin_memory,
    )

    # Sample scenes.
    # Need to rebuild the experiment for this to work.
    experiment = build_experiment(cfg, ckpt_path=checkpoint_path)
    sampled_scene_batches = experiment.exec_task("predict", dataloader=dataloader)
    sampled_scenes = torch.cat(sampled_scene_batches, dim=0)

    if cfg.get("save_pickle", False):
        # Save the sampled scenes as a pickle file.
        logging.info("Saving sampled scenes as pickle instead of rendering.")
        sampled_scenes_np = sampled_scenes.detach().cpu().numpy()
        scene_dict = {
            "scenes": sampled_scenes_np,
            "scene_vec_desc": algo.scene_vec_desc,
        }

        # Log to wandb.
        pickle_path = output_dir / "sampled_scenes.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(scene_dict, f)
        logging.info(f"Saved sampled scenes to {pickle_path}")
        wandb.save(pickle_path)
    else:
        # Render scenes.
        logging.info("Rendering sampled scenes")
        renders = get_scene_renders(
            scenes=sampled_scenes,
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
        for render in renders:
            wandb.log({"sampled_scenes": wandb.Image(render)})

    logging.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
