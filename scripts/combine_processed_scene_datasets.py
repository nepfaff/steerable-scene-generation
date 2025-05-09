"""
Script for combining multiple processed HF scene datasets into a single dataset.

This script supports combining datasets with different formats by unifying their
scene representations and creating a joint normalizer. It also stores metadata about
the subdatasets, including their ranges, names, and paths.
"""

import argparse
import gc
import logging
import os
import pathlib
import tempfile

import torch

from datasets import Dataset, Features, concatenate_datasets
from pydrake.all import PackageMap
from tqdm import tqdm

from steerable_scene_generation.algorithms.common.dataclasses import (
    RotationParametrization,
    SceneVecDescription,
)
from steerable_scene_generation.utils.hf_dataset import (
    load_hf_dataset_metadata,
    load_hf_dataset_with_metadata,
    normalize_all_scenes,
    save_hf_dataset_with_metadata,
    unnormalize_all_scenes,
)
from steerable_scene_generation.utils.min_max_scaler import (
    MinMaxScaler,
    fit_normalizer_hf,
)


def update_scenes(
    dataset: Dataset,
    scene_vec_desc: SceneVecDescription,
    new_scene_vec_desc: SceneVecDescription,
    new_max_num_objects_per_scene: int,
    num_procs: int = os.cpu_count(),
    chunk_size: int = 1_000_000,
    batch_size: int = 1000,
) -> Dataset:
    """
    Updates the scenes in the dataset to match the new scene vector description.
    """
    dataset.set_format("torch")

    def update_scenes_batch(batch: dict) -> dict:
        scenes = batch["scenes"]  # Shape (num_scenes, num_objects, scene_vec_len)
        assert (
            len(scenes.shape) == 3
        ), f"Expected scenes to be of shape (N, O, V), got {scenes.shape}"

        # Convert scenes to dict form.
        scenes_dict_form = []
        for scene in scenes:
            scene_dict_form = []
            for obj in scene:
                translation = scene_vec_desc.get_translation_vec(obj)
                # Get rotation as quaternion for unified representation
                quaternion = scene_vec_desc.get_quaternion(obj)
                model_path = scene_vec_desc.get_model_path(obj)
                if model_path is None:
                    # Skip empty objects.
                    continue
                obj_dict = {
                    "translation": translation.tolist(),
                    "quaternion": quaternion.tolist(),
                    "model_path": model_path,
                }
                scene_dict_form.append(obj_dict)
            scenes_dict_form.append(scene_dict_form)

        # Convert to new scene format.
        updated_scenes = []
        for scene in scenes_dict_form:
            updated_scene = []
            for obj in scene:
                model_path = obj["model_path"]
                # This includes the [empty] object at the last slot.
                model_path_vec = torch.zeros(len(new_scene_vec_desc.model_paths) + 1)
                model_path_vec[new_scene_vec_desc.model_paths.index(model_path)] = 1

                # Convert quaternion to the target rotation parametrization.
                quaternion_tensor = torch.tensor(obj["quaternion"])
                rotation_vec = new_scene_vec_desc.quaternion_to_rotation_vec(
                    quaternion_tensor
                )

                # Translation is always 3D.
                translation_tensor = torch.tensor(obj["translation"])

                updated_obj = new_scene_vec_desc.get_scene_or_obj_from_components(
                    translation_vec=translation_tensor,
                    rotation_vec=rotation_vec,
                    model_path_vec=model_path_vec,
                )
                updated_scene.append(updated_obj)

            if len(updated_scene) < new_max_num_objects_per_scene:
                # Pad with zeros. Represent [empty] objects as the last category.
                empty_vec = torch.zeros_like(updated_scene[0])
                empty_vec[-1] = 1.0
                updated_scene += [empty_vec] * (
                    new_max_num_objects_per_scene - len(updated_scene)
                )

            updated_scenes.append(torch.stack(updated_scene))

        batch["scenes"] = torch.stack(updated_scenes)
        return batch

    # For extremely large datasets, process in chunks and save intermediate results.
    if len(dataset) > chunk_size:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Process dataset in chunks.
            output_paths = []
            for i in tqdm(range(0, len(dataset), chunk_size), desc="Processing chunks"):
                end_idx = min(i + chunk_size, len(dataset))
                chunk = dataset.select(range(i, end_idx))

                # Process this chunk.
                updated_chunk = chunk.map(
                    update_scenes_batch,
                    num_proc=num_procs,
                    desc=f"Updating scenes (chunk {i//chunk_size + 1} of "
                    f"{len(dataset)//chunk_size + 1})",
                    batched=True,
                    batch_size=batch_size,
                )

                # Save this chunk to disk.
                chunk_path = os.path.join(temp_dir, f"chunk_{i//chunk_size}.arrow")
                updated_chunk.save_to_disk(chunk_path, num_proc=num_procs)
                output_paths.append(chunk_path)

                # Clean up memory.
                del chunk
                del updated_chunk
                gc.collect()

            # Load and combine all chunks.
            logging.info("Combining processed chunks...")
            updated_datasets = [
                Dataset.load_from_disk(path, keep_in_memory=False)
                for path in output_paths
            ]
            logging.info("Concatenating datasets...")
            updated_dataset = concatenate_datasets(updated_datasets)
    else:
        updated_dataset = dataset.map(
            update_scenes_batch,
            num_proc=num_procs,
            desc="Updating scenes",
            batched=True,
            batch_size=batch_size,
        )

    # Set the features for the updated dataset.
    scene_shape = updated_dataset[0]["scenes"].shape
    # Get existing features and update only the 'scenes' feature.
    existing_features = updated_dataset.features
    features_dict = existing_features.to_dict()
    features_dict["scenes"] = {
        "_type": "Sequence",
        "feature": {
            "_type": "Sequence",
            "feature": {"_type": "Value", "dtype": "float32"},
            "length": scene_shape[1],
        },
        "length": scene_shape[0],
    }
    features = Features.from_dict(features_dict)
    # This often fails with multiprocessing.
    updated_dataset = updated_dataset.cast(features, num_proc=1)
    return updated_dataset


def update_scenes_and_save(
    dataset: Dataset,
    scene_vec_desc: SceneVecDescription,
    new_scene_vec_desc: SceneVecDescription,
    new_max_num_objects_per_scene: int,
    num_procs: int,
    temp_dir: str,
    dataset_idx: int,
) -> str:
    """
    Updates the scenes in the dataset to match the new scene vector description
    and saves the result to disk to minimize memory usage.

    Args:
        dataset (Dataset): The dataset to update.
        scene_vec_desc (SceneVecDescription): The original scene vector description.
        new_scene_vec_desc (SceneVecDescription): The new scene vector description.
        new_max_num_objects_per_scene (int): The maximum number of objects per scene.
        num_procs (int): Number of processes to use for parallel processing.
        temp_dir (str): Directory to save the temporary file.
        dataset_idx (int): The index of the dataset.

    Returns:
        str: Path to the saved dataset.
    """
    dataset.set_format("torch")

    # Update the scenes.
    updated_dataset = update_scenes(
        dataset,
        scene_vec_desc=scene_vec_desc,
        new_scene_vec_desc=new_scene_vec_desc,
        new_max_num_objects_per_scene=new_max_num_objects_per_scene,
        num_procs=num_procs,
    )

    # Save to disk.
    output_path = os.path.join(temp_dir, f"updated_dataset_{dataset_idx}.arrow")
    updated_dataset.save_to_disk(output_path, num_proc=num_procs)

    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_files",
        nargs="+",
        type=str,
        required=True,
        help="List of input HF files to combine.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output file to save the combined dataset to.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        type=int,
        required=False,
        default=None,
        help="A label for each of the datasets.",
    )
    parser.add_argument(
        "--subdataset_names",
        nargs="+",
        type=str,
        required=False,
        default=None,
        help="Names for each subdataset. If not provided, will use the basename of the "
        "input files.",
    )
    parser.add_argument(
        "--target_rotation_parametrization",
        type=str,
        choices=["axis_angle", "procrustes", "quaternion"],
        default=None,
        help="Target rotation parametrization for the unified dataset. If not "
        "provided, will use the first dataset's parametrization.",
    )
    parser.add_argument(
        "--not_normalize_one_hot_features",
        action="store_true",
        help="If true, will not normalize the one-hot features. Note that this needs "
        "to be enabled, even if all input datasets had their one-hot vector "
        "unnormalized.",
    )
    parser.add_argument(
        "--dataset_idx",
        type=int,
        default=None,
        help="Index (starting at zero) of the dataset to process. If not provided, "
        "will process all datasets. If specified, then only the selected dataset will"
        "be converted into the combined target format but the datasets won't be "
        "combined. One would normally do this for all datasets separately and then "
        "concatenate them in the end. This is useful for huge datasets.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count(),
        help="Number of workers for parallel processing. Defaults to all CPU cores.",
    )
    args = parser.parse_args()
    input_paths = args.input_files
    output_path = args.output_file
    labels = args.labels
    subdataset_names = args.subdataset_names
    target_rotation_parametrization = args.target_rotation_parametrization
    not_normalize_one_hot_features = args.not_normalize_one_hot_features
    dataset_idx = args.dataset_idx
    num_workers = args.num_workers

    # Validate inputs.
    assert labels is None or len(input_paths) == len(labels)
    assert dataset_idx is None or (0 <= dataset_idx < len(input_paths)), (
        f"Dataset index {dataset_idx} is out of range for {len(input_paths)} input "
        "files."
    )

    if subdataset_names is None:
        # Use basenames of input files as subdataset names.
        subdataset_names = [pathlib.Path(path).stem for path in input_paths]
    else:
        assert len(subdataset_names) == len(
            input_paths
        ), "Number of subdataset names must match number of input files"

    unnormalized_datasets = []
    scene_vec_descs = []
    combined_model_paths = set()
    rotation_parametrizations = []
    max_num_objects_per_scene = 0
    mean_num_objects_per_scene = None
    welded_object_model_paths = set()

    # First pass: collect metadata from all datasets.
    for i, input_path in enumerate(input_paths):
        logging.info(
            f"Collecting metadata for dataset {i+1} of {len(input_paths)}: {input_path}"
        )

        # Load dataset and metadata.
        metadata = load_hf_dataset_metadata(input_path)

        # Collect format information.
        rotation_parametrizations.append(metadata["rotation_parametrization"])

        # Collect model paths.
        combined_model_paths.update(metadata["model_paths"])
        if "welded_object_model_paths" in metadata:
            welded_object_model_paths.update(metadata["welded_object_model_paths"])

        # Update max objects per scene.
        if metadata["max_num_objects_per_scene"] > max_num_objects_per_scene:
            max_num_objects_per_scene = metadata["max_num_objects_per_scene"]

    # Determine target format.
    target_rotation_param = (
        target_rotation_parametrization or rotation_parametrizations[0]
    )

    logging.info(f"Unifying datasets with the following target format:")
    logging.info(f"  Rotation parametrization: {target_rotation_param}")
    logging.info(f"  Combined model paths length: {len(combined_model_paths)}")
    logging.info(
        f"  Combined welded object model paths length: {len(welded_object_model_paths)}"
    )
    logging.info(
        f"  Combined max number of objects per scene: {max_num_objects_per_scene}."
    )
    logging.info(
        f"  Not normalizing one-hot features: {not_normalize_one_hot_features}"
    )

    # Track subdataset ranges.
    subdataset_ranges = []
    current_index = 0

    # Second pass: process each dataset.
    for i, input_path in enumerate(tqdm(input_paths, desc="Processing datasets")):
        if dataset_idx is not None and i != dataset_idx:
            logging.info(f"Skipping dataset {i+1} of {len(input_paths)}: {input_path}")
            continue
        logging.info(f"Processing dataset {i+1} of {len(input_paths)}: {input_path}")

        # Load dataset and metadata.
        dataset, metadata = load_hf_dataset_with_metadata(input_path)

        # Track subdataset range
        subdataset_ranges.append((current_index, current_index + len(dataset)))
        current_index += len(dataset)

        # Load normalizer.
        normalizer = MinMaxScaler(output_min=-1.0, output_max=1.0, clip=True)
        normalizer.load_serializable_state(metadata["normalizer_state"])
        if not normalizer.is_fitted:
            raise ValueError(f"Normalizer for dataset {input_path} is not fitted!")

        # Unnormalize the dataset.
        unnormalized_dataset = unnormalize_all_scenes(
            normalizer=normalizer, hf_dataset=dataset, num_procs=num_workers
        )

        scene_vec_desc = SceneVecDescription(
            drake_package_map=PackageMap(),
            static_directive=None,
            translation_vec_len=metadata["translation_vec_len"],
            rotation_parametrization=RotationParametrization.from_str(
                metadata["rotation_parametrization"]
            ),
            model_paths=metadata["model_paths"],
            model_path_vec_len=metadata["model_path_vec_len"],
            welded_object_model_paths=metadata.get("welded_object_model_paths", []),
        )
        scene_vec_descs.append(scene_vec_desc)

        # Update mean number of objects per scene.
        if mean_num_objects_per_scene is None:
            mean_num_objects_per_scene = (
                metadata["mean_num_objects_per_scene"]
                if "mean_num_objects_per_scene" in metadata
                else None
            )
        elif metadata.get("mean_num_objects_per_scene") is not None:
            mean_num_objects_per_scene += metadata["mean_num_objects_per_scene"]

        # Add labels if specified.
        if labels is not None:
            unnormalized_dataset = unnormalized_dataset.add_column(
                "labels", [labels[i]] * len(unnormalized_dataset)
            )

        # Add subdataset index.
        unnormalized_dataset = unnormalized_dataset.add_column(
            "subdataset_idx", [i] * len(unnormalized_dataset)
        )

        unnormalized_datasets.append(unnormalized_dataset)

    mean_num_objects_per_scene = (
        mean_num_objects_per_scene / len(input_paths)
        if mean_num_objects_per_scene is not None
        else None
    )
    combined_model_paths = list(combined_model_paths)
    welded_object_model_paths = list(welded_object_model_paths)

    # Create the new scene vector description.
    new_scene_vec_desc = SceneVecDescription(
        drake_package_map=PackageMap(),
        static_directive=None,
        translation_vec_len=3,
        rotation_parametrization=RotationParametrization.from_str(
            target_rotation_param
        ),
        model_paths=combined_model_paths,
        model_path_vec_len=len(combined_model_paths) + 1,  # +1 for [empty]
        welded_object_model_paths=welded_object_model_paths,
    )

    # Update the scene vectors.
    updated_dataset_paths = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, (dataset, scene_vec_desc) in enumerate(
            tqdm(
                zip(unnormalized_datasets, scene_vec_descs),
                desc="Updating scene datasets",
                total=len(unnormalized_datasets),
            )
        ):
            logging.info(
                f"Updating scene dataset {i+1} of {len(unnormalized_datasets)}"
            )
            dataset_path = update_scenes_and_save(
                dataset=dataset,
                scene_vec_desc=scene_vec_desc,
                new_scene_vec_desc=new_scene_vec_desc,
                new_max_num_objects_per_scene=max_num_objects_per_scene,
                num_procs=num_workers,
                temp_dir=temp_dir,
                dataset_idx=i,
            )
            updated_dataset_paths.append(dataset_path)

            # Clear the original dataset from memory.
            del dataset

        # Load and combine datasets from disk.
        logging.info("Loading and combining datasets from disk...")
        updated_datasets = [
            Dataset.load_from_disk(path, keep_in_memory=False)
            for path in tqdm(updated_dataset_paths, desc="Loading updated datasets")
        ]

        if dataset_idx is None:
            logging.info("Combining datasets...")
            combined_unnormalized = concatenate_datasets(updated_datasets)
        else:
            combined_unnormalized = updated_datasets[0]

        # Convert to torch format and fit new normalizer to combined dataset.
        combined_unnormalized.set_format("torch", columns=["scenes"])
        logging.info("Fitting new normalizer...")
        new_normalizer, new_normalizer_state = fit_normalizer_hf(
            hf_dataset=combined_unnormalized, num_proc=num_workers
        )

        new_model_path_vec_len = len(combined_model_paths) + 1  # +1 for [empty]
        if not_normalize_one_hot_features:
            logging.info("Not normalizing one-hot features...")

            # Exclude the one-hot model path vector from the normalizer. Assumes that
            # the model path vector comes last.
            new_normalizer.params["scale"][-new_model_path_vec_len:] = 1.0
            new_normalizer.params["min"][-new_model_path_vec_len:] = 0.0

            # Get the updated normalizer state.
            new_normalizer_state = new_normalizer.get_serializable_state()

        # Normalize the combined dataset.
        combined_normalized = normalize_all_scenes(
            normalizer=new_normalizer,
            hf_dataset=combined_unnormalized,
            num_procs=num_workers,
            batch_size=1,
        )

        # Create combined metadata.
        combined_metadata = {
            "rotation_parametrization": target_rotation_param,
            "translation_vec_len": 3,
            "model_path_vec_len": new_model_path_vec_len,
            "max_num_objects_per_scene": max_num_objects_per_scene,
            "mean_num_objects_per_scene": mean_num_objects_per_scene,
            "model_paths": list(combined_model_paths),
            "welded_object_model_paths": list(welded_object_model_paths),
            "normalizer_state": new_normalizer_state,
            "subdataset_ranges": subdataset_ranges,
            "subdataset_names": subdataset_names,
            "subdataset_paths": input_paths,
            "is_one_hot_vector_normalized": not not_normalize_one_hot_features,
        }

        # Save the combined dataset.
        save_hf_dataset_with_metadata(
            hf_dataset=combined_normalized,
            metadata=combined_metadata,
            dataset_path=output_path,
            num_procs=num_workers,
        )

        logging.info(f"Combined dataset info:")
        logging.info(f"Number of scenes: {len(combined_normalized)}")
        if labels is not None:
            logging.info(
                f"Number of unique labels: {len(set(combined_normalized['labels']))}"
            )
        logging.info(f"Number of subdatasets: {len(subdataset_names)}")
        for i, (name, (start, end)) in enumerate(
            zip(subdataset_names, subdataset_ranges)
        ):
            logging.info(
                f"  Subdataset {i}: {name} - {end-start} scenes (range: {start}-{end})"
            )
        logging.info(f"Saved combined dataset to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
