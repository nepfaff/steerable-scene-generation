"""
Script for converting a processed pickle dataset as produced by
`preprocess_greg_scene_data.py` to a Hugging Face dataset for training.
"""

import argparse
import os
import pickle
import tempfile

import numpy as np
import torch

from datasets import Dataset, Features, Sequence, Value, concatenate_datasets
from tqdm import tqdm

from steerable_scene_generation.utils.hf_dataset import (
    normalize_all_scenes,
    save_hf_dataset_with_metadata,
)
from steerable_scene_generation.utils.min_max_scaler import fit_normalizer


def convert_pickle_to_hf_with_metadata(
    pickle_path: str,
    output_path: str,
    chunk_size: int = 65536,
    num_workers: int = os.cpu_count(),
):
    """
    Converts a pickle dataset to Hugging Face dataset format, preserving metadata.
    Processes the data in chunks to reduce memory usage.

    Args:
        pickle_path (str): Path to the input pickle file.
        output_path (str): Path to the output directory for the Hugging Face dataset.
        chunk_size (int): Number of samples to process in a single chunk.
    """
    # Load data from pickle.
    with open(pickle_path, "rb") as f:
        pkl_data: dict = pickle.load(f)

    scenes = pkl_data["scenes"]
    labels = pkl_data.get("labels", None)

    # Filter scenes with translation above 5m in any direction. This is required as
    # some of the pickle files contain such bad data that would cause the normalizer
    # to fail.
    num_original_scenes = len(scenes)
    scenes = [scene for scene in scenes if np.all(np.abs(scene[:, :3]) <= 5.0)]
    print("Num scenes filtered:", num_original_scenes - len(scenes))

    # Fit normalizer to scenes.
    scenes_tensor = torch.from_numpy(np.asarray(scenes, dtype=np.float32))
    normalizer, normalizer_state = fit_normalizer(scenes_tensor)

    # Extract metadata.
    metadata = {
        "rotation_parametrization": pkl_data["rotation_parametrization"],
        "translation_vec_len": pkl_data["translation_vec_len"],
        "model_path_vec_len": pkl_data["model_path_vec_len"],
        "max_num_objects_per_scene": pkl_data["max_num_objects_per_scene"],
        "mean_num_objects_per_scene": pkl_data.get("mean_num_objects_per_scene", None),
        "model_paths": pkl_data["model_paths"],
        "normalizer_state": normalizer_state,
        "welded_object_model_paths": pkl_data.get("welded_object_model_paths", []),
    }

    # Verify that labels (if present) match the number of scenes.
    if labels is not None and len(scenes) != len(labels):
        raise ValueError(
            "The length of `labels` does not match the number of `scenes`."
        )

    num_objects, num_features = scenes[0].shape
    features_dict = {
        "scenes": Sequence(
            Sequence(Value("float32"), length=num_features), length=num_objects
        ),
    }
    if labels is not None:
        features_dict["labels"] = Value("int64")
    features = Features(features_dict)

    with tempfile.TemporaryDirectory() as temp_dir:
        chunk_files = []
        for start_idx in tqdm(
            range(0, len(scenes), chunk_size), desc="Processing chunks"
        ):
            chunk_scenes = scenes[start_idx : start_idx + chunk_size]
            chunk_labels = (
                labels[start_idx : start_idx + chunk_size] if labels else None
            )

            data = {"scenes": [scene.tolist() for scene in chunk_scenes]}
            if chunk_labels is not None:
                data["labels"] = chunk_labels

            chunk_dataset = Dataset.from_dict(data, features=features)
            chunk_dataset = normalize_all_scenes(
                normalizer=normalizer,
                hf_dataset=chunk_dataset,
                num_procs=num_workers,
                batch_size=1,
            )

            chunk_file = f"{temp_dir}/chunk_{start_idx}.arrow"
            chunk_dataset.save_to_disk(chunk_file)
            chunk_files.append(chunk_file)

        # Load and concatenate all chunks into the final dataset.
        chunk_datasets = [
            Dataset.load_from_disk(file, keep_in_memory=False) for file in chunk_files
        ]
        hf_dataset = concatenate_datasets(chunk_datasets)

    save_hf_dataset_with_metadata(
        hf_dataset=hf_dataset,
        metadata=metadata,
        dataset_path=output_path,
        num_procs=num_workers,
    )
    print(f"Dataset and metadata saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a pickle dataset to Hugging Face dataset format."
    )
    parser.add_argument(
        "pickle_path", type=str, help="Path to the input processed datasetpickle file."
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to the output directory for the Hugging Face dataset.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count(),
        help="Number of workers to use for the conversion.",
    )

    args = parser.parse_args()

    # Convert the dataset.
    convert_pickle_to_hf_with_metadata(
        args.pickle_path, args.output_path, num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()
