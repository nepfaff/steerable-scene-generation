"""
Script for sampling from a model with inference-time search. Inference-time search
produces a single final scene and a history of intermediate scenes.

Usage:
- Specify `load=...` to load the run ID or local path to the checkpoint. This is
    required.
- Specify `+render_history=True` to render the history.
- See `configurations/algorithm/scene_diffuser_base.yaml`/`predict` for
    configuration options.
- You can generate output variety using `experiment.seed=...`.
"""

import logging
import os
import pickle

from pathlib import Path

import graphviz
import hydra
import numpy as np
import torch
import wandb

from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict
from pydrake.all import ModelDirectives, yaml_dump_typed

from steerable_scene_generation.algorithms.scene_diffusion.scene_diffuser_base import (
    SceneDiffuserBase,
)
from steerable_scene_generation.experiments import build_experiment
from steerable_scene_generation.utils.ckpt_utils import (
    download_latest_or_best_checkpoint,
    is_run_id,
)
from steerable_scene_generation.utils.distributed_utils import is_rank_zero
from steerable_scene_generation.utils.file_parsing import get_scene_directives
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

# Set multiprocessing start method to 'spawn' to avoid CUDA initialization issues.
torch.multiprocessing.set_start_method("spawn", force=True)


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

    # Set predict mode.
    cfg.algorithm.predict.do_sample = False
    cfg.algorithm.predict.do_rearrange = False
    cfg.algorithm.predict.do_complete = False
    cfg.algorithm.predict.do_inference_time_search = True
    cfg.algorithm.predict.do_sample_scenes_with_k_closest_training_examples = False

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
        f"inference_time_search_visualization_{load_id} "
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

    # Build a fake dataloader. We need one.
    num_scenes = cfg.algorithm.predict.inference_time_search.mcts.branching_factor
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

    # Sample a single scene with inference-time search.
    # Need to rebuild the experiment for this to work.
    experiment = build_experiment(cfg, ckpt_path=checkpoint_path)
    output_batches = experiment.exec_task("predict", dataloader=dataloader)
    assert len(output_batches) == 1
    output = output_batches[0]
    history = output["history"]  # Shape (T, N, V)

    # Log the scenes for later use.
    logging.info("Logging best scenes in directives and pickle format")
    best_scene_directives = get_scene_directives(
        scenes=output["best_scenes"], scene_vec_desc=algo.scene_vec_desc
    )
    # Create directory for scene directives.
    scene_directives_dir = output_dir / "scene_directives"
    scene_directives_dir.mkdir(exist_ok=True)
    for i, drake_directive in enumerate(best_scene_directives):
        directive_path = scene_directives_dir / f"best_scene_directives_{i}.dmd.yaml"
        yaml_dump_typed(
            drake_directive, filename=str(directive_path), schema=ModelDirectives
        )
        wandb.save(str(directive_path))
    best_scenes_dict = {
        "scenes": output["best_scenes"].cpu().numpy(),
        "scene_vec_desc": algo.scene_vec_desc,
    }
    # Save pickle file to output directory.
    pickle_path = output_dir / "best_scenes_dict.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(best_scenes_dict, f)
    wandb.save(str(pickle_path))
    # Save metadata to output directory.
    metadata_path = output_dir / "metadata.pkl"
    metadata = {
        "best_costs": output["best_costs"] if "best_costs" in output else None,
        "best_scene_indices": (
            output["best_scene_indices"] if "best_scene_indices" in output else None
        ),
        "empty_object_numbers": (
            output["empty_object_numbers"] if "empty_object_numbers" in output else None
        ),
        "non_empty_object_numbers": (
            output["non_empty_object_numbers"]
            if "non_empty_object_numbers" in output
            else None
        ),
        "non_static_equilibrium_object_numbers": (
            output["non_static_equilibrium_object_numbers"]
            if "non_static_equilibrium_object_numbers" in output
            else None
        ),
        "total_penetration_distances": (
            output["total_penetration_distances"]
            if "total_penetration_distances" in output
            else None
        ),
    }
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    wandb.save(str(metadata_path))

    # Render scenes.
    render_history = cfg.get("render_history", False)
    if not render_history:
        logging.info("Not rendering scene history")
        # Always render the first ten scenes.
        history_to_render = history[:10]
    else:
        history_to_render = history
        logging.info("Rendering scene history")
    history_renders = get_scene_renders(
        scenes=history_to_render,
        scene_vec_desc=algo.scene_vec_desc,
        camera_poses=algo.cfg.visualization.camera_pose,
        camera_width=algo.cfg.visualization.image_width,
        camera_height=algo.cfg.visualization.image_height,
        background_color=algo.cfg.visualization.background_color,
        num_workers=algo.cfg.visualization.num_workers,
        use_blender_server=algo.cfg.visualization.use_blender_server,
        blender_server_url=algo.cfg.visualization.blender_server_url,
    )
    logging.info("Rendering best scenes")
    # Limit to 10 as they tend to be very similar.
    best_scene_renders = get_scene_renders(
        scenes=output["best_scenes"][:10],
        scene_vec_desc=algo.scene_vec_desc,
        camera_poses=algo.cfg.visualization.camera_pose,
        camera_width=algo.cfg.visualization.image_width,
        camera_height=algo.cfg.visualization.image_height,
        background_color=algo.cfg.visualization.background_color,
        num_workers=algo.cfg.visualization.num_workers,
        use_blender_server=algo.cfg.visualization.use_blender_server,
        blender_server_url=algo.cfg.visualization.blender_server_url,
    )

    wandb.log(
        {
            "num_iters_used": output["num_iters_used"],
            "reached_max_iters": 1.0 if output["reached_max_iters"] else 0.0,
            "first_ten_scenes": [
                wandb.Image(render) for render in history_renders[:10]
            ],
            "best_scenes": [wandb.Image(render) for render in best_scene_renders],
            "num_best_scenes": len(best_scene_renders),
        }
    )

    # Log images sequentially to enable step-by-step visualization in WandB.
    for i in range(len(history)):
        data = {
            "scene_history": (
                wandb.Image(history_renders[i]) if render_history else None
            ),
        }
        if "total_penetration_distances" in output and i < len(
            output["total_penetration_distances"]
        ):
            data["total_penetration_distance"] = output["total_penetration_distances"][
                i
            ]
        if "empty_object_numbers" in output and i < len(output["empty_object_numbers"]):
            data["empty_object_numbers"] = output["empty_object_numbers"][i]
        if "non_empty_object_numbers" in output and i < len(
            output["non_empty_object_numbers"]
        ):
            data["non_empty_object_numbers"] = output["non_empty_object_numbers"][i]
        if "non_static_equilibrium_object_numbers" in output and i < len(
            output["non_static_equilibrium_object_numbers"]
        ):
            data["non_static_equilibrium_object_numbers"] = output[
                "non_static_equilibrium_object_numbers"
            ][i]
        if "best_scene_indices" in output and i < len(output["best_scene_indices"]):
            data["best_scene_indices"] = output["best_scene_indices"][i]
        if "best_costs" in output and i < len(output["best_costs"]):
            data["best_costs"] = output["best_costs"][i]
        wandb.log(data)

    if "tree_data" in output:
        nodes = output["tree_data"]["nodes"]
        edges = output["tree_data"]["edges"]

        # Create a directed graph.
        dot = graphviz.Digraph(comment="MCTS Search Tree", format="svg")
        dot.attr("node", shape="circle")

        # Determine cost range for normalization.
        costs = [node["cost"] for node in nodes] if nodes else []
        max_cost = max(costs) if costs else 1
        min_cost = min(costs) if costs else 0

        # Add nodes with color scaling based on cost.
        for node in nodes:
            normalized_cost = (
                (node["cost"] - min_cost) / (max_cost - min_cost)
                if max_cost > min_cost
                else 0
            )

            # Red (high cost) to green (low cost).
            r = int(255 * normalized_cost)
            g = int(255 * (1 - normalized_cost))
            b = 0
            color = f"#{r:02x}{g:02x}{b:02x}"  # Hex color code

            # Determine node size - make best nodes larger.
            fontsize = "20" if node["cost"] == min_cost else "10"

            # Add node to graphviz.
            dot.node(
                str(node["id"]),
                label=f"Cost: {node['cost']:.2f}\nValue: {node['value']:.2f}\nVisits: {node['visits']}",
                style="filled",
                fillcolor=color,
                fontsize=fontsize,
            )

        # Add edges.
        for edge in edges:
            dot.edge(str(edge["from"]), str(edge["to"]))

        # Save to a temporary file and read back.
        temp_filename = str(output_dir / "mcts_tree")
        dot.render(temp_filename, cleanup=True)

        # Log the SVG file directly as an artifact.
        svg_artifact = wandb.Artifact("mcts_tree", type="visualization")
        svg_artifact.add_file(temp_filename + ".svg")
        wandb.log_artifact(svg_artifact)

        # Add the SVG file to the WandB log.
        # Note that this is a bit slow and WandB might take a while to display any logs
        # after the upload is complete.
        wandb.log({"MCTS Search Tree": wandb.Html(open(temp_filename + ".svg").read())})

    logging.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
