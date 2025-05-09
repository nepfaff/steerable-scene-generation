"""
Renders scene samples from a pickle file as semantic label images for use in metrics.
"""

import argparse
import multiprocessing
import os
import pickle

import numpy as np

from PIL import Image
from pydrake.all import RigidTransform, RollPitchYaw
from tqdm import tqdm

from steerable_scene_generation.algorithms.common.dataclasses import SceneVecDescription
from steerable_scene_generation.utils.drake_utils import make_package_map
from steerable_scene_generation.utils.logging import filter_drake_vtk_warning
from steerable_scene_generation.utils.visualization import (
    get_scene_label_image_renders,
    setup_virtual_display_if_needed,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sample_pickle_path",
        type=str,
        help="Path to the sample pickle file as saved by the scene diffuser trainer.",
    )
    parser.add_argument(
        "--output_dir",
        default="semantic_renders",
        type=str,
        help="The path to the directory to save the renders to.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of workers to use for rendering.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Number of scenes to render before saving.",
    )
    parser.add_argument(
        "--package_name",
        type=str,
        default="tri",
        help="An optional package name for resolving the model paths.",
    )
    parser.add_argument(
        "--package_file_path",
        type=str,
        default="data/tri/package.xml",
        help="An optional package file path to `package.xml` for resolving the model "
        "paths.",
    )
    args = parser.parse_args()
    sample_pickle_path = args.sample_pickle_path
    output_dir = args.output_dir
    num_workers = args.num_workers
    batch_size = args.batch_size
    package_name = args.package_name
    package_file_path = args.package_file_path

    # Load scene data.
    with open(sample_pickle_path, "rb") as f:
        scene_data = pickle.load(f)

    # Extract scene data.
    scenes: np.ndarray = scene_data["scenes"]
    scene_vec_desc: SceneVecDescription = scene_data["scene_vec_desc"]

    # Add potentially missing package map.
    if (
        package_name is not None
        and package_file_path is not None
        and not scene_vec_desc.drake_package_map.Contains(package_name)
    ):
        package_map = make_package_map(package_name, package_file_path)
        scene_vec_desc.drake_package_map.AddMap(package_map)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Render and save in batches.
    counter = 0
    scene_batches = np.array_split(scenes, np.ceil(len(scenes) / batch_size))
    for scenes in tqdm(scene_batches, desc="Rendering batches."):
        images_array = get_scene_label_image_renders(
            scenes=scenes,
            scene_vec_desc=scene_vec_desc,
            X_WC=RigidTransform(
                RollPitchYaw([-3.14, 0.0, 1.57]),
                [0.0, 0.0, 1.5],
            ),
            camera_width=640,
            camera_height=480,
            num_workers=num_workers,
        )

        # Save images to disk.
        for img_np in tqdm(images_array, desc="Saving images", leave=False):
            # Convert to uint8.
            img_normalized = (img_np * 255).clip(0, 255)
            img_uint8 = img_normalized.astype(np.uint8)

            img = Image.fromarray(img_uint8)
            path = os.path.join(output_dir, f"semantic_render_{counter:04}.png")
            img.save(path)

            counter += 1


if __name__ == "__main__":
    filter_drake_vtk_warning()
    setup_virtual_display_if_needed()
    main()
