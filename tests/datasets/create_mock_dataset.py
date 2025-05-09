"""Script for creating a Mock dataset for testing."""

from typing import Tuple

import numpy as np
import torch

from datasets import Dataset

from steerable_scene_generation.utils.hf_dataset import (
    normalize_all_scenes,
    save_hf_dataset_with_metadata,
)
from steerable_scene_generation.utils.min_max_scaler import MinMaxScaler


def fit_normalizer(scenes: torch.Tensor) -> Tuple[MinMaxScaler, dict]:
    """Fits a normalizer to the scenes and returns its state."""
    normalizer = MinMaxScaler(output_min=-1.0, output_max=1.0, clip=True)

    # Flatten scenes into object vectors and normalize separately across each object
    # vector feature.
    normalizer.fit(scenes.reshape(-1, scenes.shape[-1]))

    state = normalizer.get_serializable_state()
    return normalizer, state


def main():
    # Set random seed for reproducibility.
    np.random.seed(42)
    torch.manual_seed(42)

    # Create scenes that consist of 2 objects with random translations.
    raw_scenes = []
    language_annotations = []
    for _ in range(10):
        raw_scene = []
        for _ in range(2):
            model_path = np.random.choice(
                [
                    "tests/models/box.sdf",
                    "tests/models/sphere.sdf",
                    "tests/models/cylinder.sdf",
                ]
            )
            translation = np.random.uniform(-5.0, 5.0, size=3)
            raw_scene.append({"model_path": model_path, "translation": translation})
        raw_scenes.append(raw_scene)

        # Create language annotations.
        num_boxes = sum(["box" in obj["model_path"] for obj in raw_scene])
        num_spheres = sum(["sphere" in obj["model_path"] for obj in raw_scene])
        num_cylinders = sum(["cylinder" in obj["model_path"] for obj in raw_scene])
        language_annotations.append(
            f"{num_boxes} boxes and {num_spheres} spheres and {num_cylinders} cylinders."
        )

    model_paths = [
        "tests/models/box.sdf",
        "tests/models/sphere.sdf",
        "tests/models/cylinder.sdf",
    ]
    welded_object_model_paths = ["tests/models/cylinder.sdf"]

    # Convert to processed scene format.
    scenes = []
    for raw_scene in raw_scenes:
        scene = []
        for obj in raw_scene:
            translation = obj["translation"]
            rotation = np.eye(3).flatten()

            # One hot model path vec. This includes the [empty] object at the last
            # slot.
            model_path = obj["model_path"]
            model_path_vec = [0] * (len(model_paths) + 1)  # Shape (O+1,)
            model_path_vec[model_paths.index(model_path)] = 1

            scene.append(np.concatenate((translation, rotation, model_path_vec)))

        scenes.append(scene)

    # Fit normalizer to scenes.
    scenes_tensor = torch.from_numpy(np.asarray(scenes, dtype=np.float32))
    normalizer, normalizer_state = fit_normalizer(scenes_tensor)

    # Construct the metadata.
    metadata = {
        "rotation_parametrization": "procrustes",
        "translation_vec_len": 3,
        "model_path_vec_len": len(model_paths) + 1,  # +1 for the [empty] object
        "max_num_objects_per_scene": 2,
        "mean_num_objects_per_scene": 2,
        "model_paths": model_paths,
        "normalizer_state": normalizer_state,
        "welded_object_model_paths": welded_object_model_paths,
    }

    # Create the HF dataset with language annotations.
    hf_dataset = Dataset.from_dict(
        {"scenes": scenes, "language_annotation": language_annotations},
    )

    # Normalize the scenes.
    hf_dataset = normalize_all_scenes(normalizer, hf_dataset)

    # Save the dataset.
    save_hf_dataset_with_metadata(hf_dataset, metadata, "tests/datasets/mock_dataset")


if __name__ == "__main__":
    main()
