"""Script for visualizing a scene from a HF dataset."""

import argparse
import json

import wandb

from tqdm import tqdm

from steerable_scene_generation.utils.hf_dataset import (
    get_scene_vec_description_from_metadata,
    load_hf_dataset_with_metadata,
)
from steerable_scene_generation.utils.min_max_scaler import MinMaxScaler
from steerable_scene_generation.utils.visualization import get_visualized_scene_htmls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scenes_data_path", type=str, help="Path to the scene HF dataset folder."
    )
    parser.add_argument(
        "--scene_idx", default=0, type=int, help="Index of the scene to visualize."
    )
    parser.add_argument(
        "--filter_string",
        type=str,
        default=None,
        help="String to filter scenes by their language_annotation field.",
    )
    parser.add_argument(
        "--not_weld_objects",
        action="store_true",
        help="Do not weld the objects to the world frame. This makes it easier to "
        + "spot penetration violations.",
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use Weights and Biases for logging."
    )
    parser.add_argument(
        "--background_color",
        default=[1.0, 1.0, 1.0],
        type=json.loads,
        help="The meshcat background color.",
    )
    parser.add_argument(
        "--visualize_proximity",
        action="store_true",
        help="Whether to visualize proximity.",
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
        help="An optional list of package file paths to `package.xml` for resolving the model "
        "paths.",
    )
    parser.add_argument(
        "--static_directive",
        type=str,
        default=None,
        help="An optional static directive for the scene that contains welded objects.",
    )
    args = parser.parse_args()
    scenes_data_path = args.scenes_data_path
    scene_idx = args.scene_idx
    filter_string = args.filter_string
    not_weld_objects = args.not_weld_objects
    background_color = args.background_color
    visualize_proximity = args.visualize_proximity
    package_names = args.package_names
    package_file_paths = args.package_file_paths
    static_directive = args.static_directive

    use_wandb = args.use_wandb
    if use_wandb:
        wandb.init(
            project="steerable_scene_generation_scene_visualization",
            name=f"visualized_scene: {str(scenes_data_path)}[{scene_idx}]",
            config=vars(args),
        )

    # Load dataset.
    hf_dataset, metadata = load_hf_dataset_with_metadata(scenes_data_path)
    hf_dataset.set_format(type="torch")

    # Apply filter if specified.
    if filter_string:
        filtered_indices = []
        for idx, data in tqdm(
            enumerate(hf_dataset),
            desc="Seaching scene with specified filter string",
            total=len(hf_dataset),
        ):
            if (
                "language_annotation" in data
                and filter_string in data["language_annotation"]
            ):
                filtered_indices.append(idx)
                if len(filtered_indices) > scene_idx:
                    break
        if not filtered_indices:
            raise ValueError(
                f"No scenes found with language_annotation containing '{filter_string}'"
            )
        scene_idx = filtered_indices[scene_idx]

    # Load the normalizer.
    normalizer = MinMaxScaler(output_min=-1.0, output_max=1.0, clip=True)
    normalizer.load_serializable_state(metadata["normalizer_state"])
    if not normalizer.is_fitted:
        raise ValueError("Normalizer is not fitted!")

    # Create the scene vector description.
    scene_vec_desc = get_scene_vec_description_from_metadata(
        metadata,
        static_directive=static_directive,
        package_names=package_names,
        package_file_paths=package_file_paths,
    )

    # Get the scene.
    scene_data = hf_dataset[scene_idx]
    scene = scene_data["scenes"][None]  # Shape (1, num_objects, num_features)

    # Unnormalize the scene.
    unnormalized_scene = normalizer.inverse_transform(
        scene.reshape(-1, scene.shape[-1])
    ).reshape(scene.shape)

    # Print language annotation if it exists.
    if "language_annotation" in scene_data:
        print("Language annotation:\n", scene_data["language_annotation"])

    # Visualize scene.
    html = get_visualized_scene_htmls(
        scenes=unnormalized_scene,
        scene_vec_desc=scene_vec_desc,
        background_color=background_color,
        simulation_time=5.0,
        weld_objects=not not_weld_objects,
        visualize_proximity=visualize_proximity,
    )[0]

    if use_wandb:
        wandb.log({"scene": wandb.Html(html)})


if __name__ == "__main__":
    main()
