"""Renders a dataset or a subset of it to wandb."""

import argparse
import multiprocessing

import torch
import wandb
import yaml

from PIL import Image

from steerable_scene_generation.utils.hf_dataset import (
    get_scene_vec_description_from_metadata,
    load_hf_dataset_with_metadata,
)
from steerable_scene_generation.utils.min_max_scaler import MinMaxScaler
from steerable_scene_generation.utils.visualization import (
    get_scene_renders,
    setup_virtual_display_if_needed,
)


def load_camera_configs():
    config_path = "./configurations/algorithm/scene_diffuser_base.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scenes_data_path",
        type=str,
        help="Path to the scene HF dataset folder.",
    )
    parser.add_argument(
        "--start_idx",
        type=float,
        default=0,
        help="Index of the first scene to render. Will be cast to int.",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=None,
        help="Number of scenes to render.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of workers to use for rendering.",
    )
    parser.add_argument(
        "--package_names",
        nargs="+",
        type=str,
        default=["tri", "gazebo", "greg"],
        help="An optional list of package names for resolving the model paths.",
    )
    parser.add_argument(
        "--package_file_paths",
        nargs="+",
        type=str,
        default=[
            "data/tri/package.xml",
            "data/gazebo/package.xml",
            "data/greg/package.xml",
        ],
        help="An optional list of package file paths to `package.xml` for resolving "
        "the model paths.",
    )
    parser.add_argument(
        "--static_directive",
        type=str,
        default=None,
        help="An optional static directive for the scene that contains welded objects.",
    )
    parser.add_argument(
        "--use_blender_server",
        type=bool,
        default=False,
        help="Whether to use the Blender server for rendering.",
    )
    args = parser.parse_args()

    # Load dataset.
    hf_dataset, metadata = load_hf_dataset_with_metadata(args.scenes_data_path)
    hf_dataset.set_format(type="torch")

    # Select scenes to render.
    start_idx = int(args.start_idx)
    num_scenes = args.num_scenes
    if num_scenes is None:
        num_scenes = len(hf_dataset) - start_idx
    dataset_slice = list(range(start_idx, start_idx + num_scenes))
    if len(dataset_slice) == 0:
        raise ValueError("No scenes to render.")

    wandb.init(
        project="steerable_scene_generation_dataset_renders",
        name=f"dataset: {args.scenes_data_path}[{start_idx}:{start_idx+num_scenes}]",
        config=vars(args),
    )

    # Load the normalizer.
    normalizer = MinMaxScaler(output_min=-1.0, output_max=1.0, clip=True)
    normalizer.load_serializable_state(metadata["normalizer_state"])
    if not normalizer.is_fitted:
        raise ValueError("Normalizer is not fitted!")

    # Create the scene vector description.
    scene_vec_desc = get_scene_vec_description_from_metadata(
        metadata,
        static_directive=args.static_directive,
        package_names=args.package_names,
        package_file_paths=args.package_file_paths,
    )

    # Get and unnormalize scenes.
    scenes = torch.stack([hf_dataset[i]["scenes"] for i in dataset_slice])
    unnormalized_scenes = normalizer.inverse_transform(
        scenes.reshape(-1, scenes.shape[-1])
    ).reshape(scenes.shape)

    # Load camera poses from config.
    config = load_camera_configs()
    camera_poses = {
        key: {
            "xyz": config["visualization"]["camera_pose"][key]["xyz"],
            "rpy": config["visualization"]["camera_pose"][key]["rpy"],
        }
        for key in config["visualization"]["camera_pose"].keys()
    }

    images_array = get_scene_renders(
        scenes=unnormalized_scenes,
        scene_vec_desc=scene_vec_desc,
        camera_width=640,
        camera_height=480,
        background_color=[1.0, 1.0, 1.0],
        num_workers=args.num_workers,
        camera_poses=camera_poses,
        use_blender_server=args.use_blender_server,
        blender_server_url=args.blender_server_url,
    )

    # Log renders to wandb.
    images = [wandb.Image(Image.fromarray(image)) for image in images_array]
    # Wandb only supports up to 108 images per list. Split into chunks of 108.
    image_chunks = [images[i : i + 108] for i in range(0, len(images), 108)]
    for i, chunk in enumerate(image_chunks):
        wandb.log({f"dataset_renders_{i}": chunk})


if __name__ == "__main__":
    setup_virtual_display_if_needed()
    main()
