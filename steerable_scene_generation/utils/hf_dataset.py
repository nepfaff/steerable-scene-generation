import json
import logging
import os

from typing import Any, Dict, List, Tuple

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from pydrake.all import PackageMap

from steerable_scene_generation.algorithms.common.dataclasses import (
    RotationParametrization,
    SceneVecDescription,
)
from steerable_scene_generation.utils.drake_utils import make_package_map
from steerable_scene_generation.utils.min_max_scaler import MinMaxScaler

logger = logging.getLogger(__name__)


def load_hf_dataset_metadata(dataset_path: str) -> Dict:
    """
    Load the metadata from the Hugging Face dataset.

    Args:
        dataset_path (str): Path to the Hugging Face dataset on disk.

    Returns:
        Dict: The metadata dictionary.
    """
    # Expand user path (e.g., ~).
    dataset_path = os.path.expanduser(dataset_path)

    # Load the metadata
    metadata_path = os.path.join(dataset_path, "metadata.json")
    with open(metadata_path, "r") as metadata_file:
        metadata = json.load(metadata_file)

    return metadata


def load_hf_dataset_with_metadata(
    dataset_path: str, keep_in_memory: bool = False
) -> Tuple[Dataset, Dict]:
    """
    Load the Hugging Face dataset and metadata.

    Args:
        dataset_path (str): Path to the Hugging Face dataset on disk or a Hugging Face
            Hub dataset ID.
        keep_in_memory (bool): Whether to copy the dataset in-memory.

    Returns:
        Tuple[Dataset, Dict]: A tuple of the loaded dataset and metadata.
    """
    # Check if the path is a Hugging Face Hub dataset ID.
    if "/" in dataset_path and not os.path.exists(dataset_path):
        # Load from Hub.
        hf_dataset = load_dataset(dataset_path, split="train")

        # Load metadata from the separate file.
        try:
            # Try to download the metadata file.
            metadata_content = hf_hub_download(
                repo_id=dataset_path,
                filename="metadata.json",
                repo_type="dataset",
                revision="main",
            )
            with open(metadata_content, "r") as f:
                metadata = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata from {dataset_path}: {str(e)}")
            raise
    else:
        # Load from local disk.
        dataset_path = os.path.expanduser(dataset_path)
        hf_dataset = Dataset.load_from_disk(dataset_path, keep_in_memory=keep_in_memory)
        metadata = load_hf_dataset_metadata(dataset_path)

    return hf_dataset, metadata


def save_hf_dataset_metadata(metadata: Dict[str, Any], dataset_path: str) -> None:
    """
    Save the metadata.

    Args:
        metadata (Dict): The metadata dictionary.
        dataset_path (str): Path to save the Hugging Face dataset on disk.
    """
    metadata_path = os.path.join(dataset_path, "metadata.json")
    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=4)


def save_hf_dataset_with_metadata(
    hf_dataset: Dataset, metadata: Dict[str, Any], dataset_path: str, num_procs: int = 1
) -> None:
    """
    Save the Hugging Face dataset and metadata.

    Args:
        hf_dataset (Dataset): The Hugging Face dataset.
        metadata (Dict): The metadata dictionary.
        dataset_path (str): Path to save the Hugging Face dataset on disk.
        num_procs (int): The number of processes to use for saving the dataset.
    """
    # Save the dataset.
    hf_dataset.save_to_disk(dataset_path, num_proc=num_procs)

    # Save the metadata.
    save_hf_dataset_metadata(metadata=metadata, dataset_path=dataset_path)


def get_scene_vec_description_from_metadata(
    metadata: Dict[str, Any],
    static_directive: str | None = None,
    package_names: List[str] | None = None,
    package_file_paths: List[str] | None = None,
) -> SceneVecDescription:
    """
    Get the scene vector description from the metadata.

    Args:
        metadata (Dict): The metadata dictionary.
        static_directive (str | None): The static directive.
        package_names (List[str] | None): The package names.
        package_file_paths (List[str] | None): The package file paths.

    Returns:
        SceneVecDescription: The scene vector description.
    """
    if package_names is None:
        package_names = []
    if package_file_paths is None:
        package_file_paths = []

    if len(package_names) != len(package_file_paths):
        raise ValueError(
            "Length of package_names and package_file_paths should be equal."
        )

    package_map = PackageMap()
    for package_name, package_file_path in zip(package_names, package_file_paths):
        package_map.AddMap(make_package_map(package_name, package_file_path))
    scene_vec_desc = SceneVecDescription(
        drake_package_map=package_map,
        static_directive=static_directive,
        translation_vec_len=metadata["translation_vec_len"],
        rotation_parametrization=RotationParametrization.from_str(
            metadata["rotation_parametrization"]
        ),
        model_paths=metadata["model_paths"],
        model_path_vec_len=metadata.get("model_path_vec_len", None),
        welded_object_model_paths=metadata.get("welded_object_model_paths", []),
    )

    return scene_vec_desc


def normalize_all_scenes(
    normalizer: MinMaxScaler,
    hf_dataset: Dataset,
    batch_size: int = 4096,
    num_procs: int = 1,
) -> Dataset:
    """
    Normalize all scenes in the dataset and update the `scenes` field in-place without
    loading the entire dataset into memory. It uses CUDA if available.

    Args:
        normalizer (MinMaxScaler): The fitted normalizer.
        hf_dataset (Dataset): The HF dataset whose "scenes" field to normalize.
        batch_size (int): The number of samples to process in a single batch.

    Returns:
        Dataset: The HF dataset with the normalized scenes.
    """
    hf_dataset.set_format("torch")

    def normalize_batch(batch: dict) -> dict:
        scenes = batch["scenes"]
        normalized_scenes = normalizer.transform(
            scenes.reshape(-1, scenes.shape[-1])
        ).reshape(scenes.shape)
        batch["scenes"] = normalized_scenes
        return batch

    # Apply normalization to the `scenes` field in batches.
    normalized_hf_dataset = hf_dataset.map(
        normalize_batch,
        batched=True if batch_size > 1 else False,
        batch_size=batch_size,
        num_proc=num_procs,
        desc="Normalizing scenes",
    )
    return normalized_hf_dataset


def unnormalize_all_scenes(
    normalizer: MinMaxScaler,
    hf_dataset: Dataset,
    batch_size: int = 4096,
    num_procs: int = 1,
) -> Dataset:
    """
    Unormalize all scenes in the dataset and update the `scenes` field in-place without
    loading the entire dataset into memory.

    Args:
        normalizer (MinMaxScaler): The fitted normalizer.
        hf_dataset (Dataset): The HF dataset whose "scenes" field to unnormalize.
        batch_size (int): The number of samples to process in a single batch.

    Returns:
        Dataset: The HF dataset with the unnormalized scenes.
    """
    hf_dataset.set_format("torch")

    def unnormalize_batch(batch: dict) -> dict:
        scenes = batch["scenes"]
        unnormalized_scenes = normalizer.inverse_transform(
            scenes.reshape(-1, scenes.shape[-1])
        ).reshape(scenes.shape)
        batch["scenes"] = unnormalized_scenes
        return batch

    # Apply unormalization to the `scenes` field in batches.
    unormalized_hf_dataset = hf_dataset.map(
        unnormalize_batch,
        batched=True if batch_size > 1 else False,
        batch_size=batch_size,
        num_proc=num_procs,
        desc="Unnormalizing scenes",
    )
    return unormalized_hf_dataset


def upload_dataset_to_hub(
    hf_dataset: Dataset,
    metadata: Dict[str, Any],
    hub_dataset_id: str,
    private: bool = False,
    token: str | None = None,
) -> None:
    """
    Upload a dataset to the Hugging Face Hub.

    Args:
        hf_dataset (Dataset): The Hugging Face dataset to upload.
        metadata (Dict[str, Any]): The dataset metadata.
        hub_dataset_id (str): The Hugging Face Hub dataset ID (e.g.,
            "username/dataset-name").
        private (bool): Whether to make the dataset private.
        token (str | None): Hugging Face API token. If None, will try to use the token
            from the environment.
    """
    # Initialize the API.
    api = HfApi(token=token)

    # Try to create the repository if it doesn't exist.
    api.create_repo(
        repo_id=hub_dataset_id, repo_type="dataset", private=private, exist_ok=True
    )

    # Push dataset to hub.
    hf_dataset.push_to_hub(hub_dataset_id, private=private, token=token)

    # Upload the metadata as a separate file.
    try:
        api.upload_file(
            path_or_fileobj=json.dumps(metadata, indent=4).encode(),
            path_in_repo="metadata.json",
            repo_id=hub_dataset_id,
            repo_type="dataset",
        )
        logger.info("Successfully uploaded metadata.json")
    except Exception as e:
        logger.error(f"Failed to upload metadata: {str(e)}")
        raise

    logger.info(f"Dataset uploaded to https://huggingface.co/datasets/{hub_dataset_id}")
