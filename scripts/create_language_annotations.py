"""Script for adding language annotations to a HF dataset."""

import argparse
import logging
import os
import random
import tempfile

from functools import partial

import torch

from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from steerable_scene_generation.algorithms.common.dataclasses import SceneVecDescription
from steerable_scene_generation.utils.hf_dataset import (
    get_scene_vec_description_from_metadata,
    load_hf_dataset_with_metadata,
    normalize_all_scenes,
    save_hf_dataset_with_metadata,
    unnormalize_all_scenes,
)
from steerable_scene_generation.utils.min_max_scaler import MinMaxScaler
from steerable_scene_generation.utils.scene_language_annotation import (
    LanguageMode,
    SceneType,
    get_language_annotation,
)


def process_chunk(
    start_idx: int,
    chunk_size: int,
    base_dataset: Dataset,
    scene_vec_desc: SceneVecDescription,
    scene_type: SceneType,
    language_mode: LanguageMode,
    num_spatial_relation_annotations: int,
    subset_probability: float,
    max_spatial_relationships: int,
    spatial_relation_distance_threshold: float,
    temp_dir: str,
) -> str:
    """
    Process a chunk of the dataset to add language annotations.

    Args:
        start_idx (int): The starting index of the chunk.
        chunk_size (int): The size of the chunk to process.
        base_dataset (Dataset): The base HF dataset.
        scene_vec_desc (SceneVecDescription): The scene vector description.
        scene_type (SceneType): The type of scene that is contained in the dataset.
        language_mode (LanguageMode): The language mode to use for annotations.
        num_spatial_relation_annotations (int): The number of spatial relation
            annotations to include per scene if the language mode is
            'spatial_relations'.
        subset_probability (float): The probability that the object name annotation will
            list a subset of the objects in the scene.
        max_spatial_relationships (int): The maximum number of spatial relationships to
            include in the annotation.
        spatial_relation_distance_threshold (float): The maximum distance between
            objects to consider for spatial relationships.
        temp_dir (str): The temporary directory to save the chunk files.

    Returns:
        str: The path to the saved chunk file.
    """
    chunk = base_dataset[start_idx : start_idx + chunk_size]

    new_rows = []
    for i in range(len(chunk["scenes"])):
        row = {key: chunk[key][i] for key in chunk}

        # Generate multiple annotations.
        language_annotations = get_language_annotation(
            scene_vec_desc=scene_vec_desc,
            scene=row["scenes"],
            scene_type=scene_type,
            language_mode=language_mode,
            num_spatial_relation_annotations=num_spatial_relation_annotations,
            subset_probability=subset_probability,
            max_spatial_relationships=max_spatial_relationships,
            spatial_relation_distance_threshold=spatial_relation_distance_threshold,
        )

        # Create a new row for each annotation.
        for language_annotation in language_annotations:
            row_dict = row.copy()
            row_dict["language_annotation"] = language_annotation
            new_rows.append(row_dict)

    # Convert chunk to a dataset and save to a temporary file.
    chunk_dataset = Dataset.from_dict(
        {key: [row[key] for row in new_rows] for key in new_rows[0].keys()}
    )
    chunk_file = f"{temp_dir}/chunk_{start_idx}.arrow"
    chunk_dataset.save_to_disk(chunk_file)

    return chunk_file


def create_new_dataset_with_language_annotations(
    base_dataset: Dataset,
    scene_vec_desc: SceneVecDescription,
    scene_type: SceneType,
    language_mode: LanguageMode,
    num_spatial_relation_annotations: int = 1,
    subset_probability: float = 0.5,
    max_spatial_relationships: int = 5,
    spatial_relation_distance_threshold: float = 0.2,
    chunk_size: int = 32768,
    max_workers: int = -1,
) -> Dataset:
    """
    Create a new dataset with language annotations. The new dataset contains the same
    scene multiple times, each with different language annotations. This implementation
    processes the dataset in chunks to minimize memory usage and uses parallel
    processing for faster annotation generation.

    Args:
        base_dataset (Dataset): The base HF dataset.
        scene_vec_desc (SceneVecDescription): The scene vector description.
        scene_type (SceneType): The type of scene that is contained in the dataset.
        language_mode (LanguageMode): The language mode to use for annotations.
        num_spatial_relation_annotations (int): The number of spatial relation
            annotations to include per scene if the language mode is
            'spatial_relations'.
        subset_probability (float): The probability that the object name annotation will
            list a subset of the objects in the scene.
        max_spatial_relationships (int): The maximum number of spatial relationships to
            include in the annotation.
        spatial_relation_distance_threshold (float): The maximum distance between
            objects to consider for spatial relationships.
        chunk_size (int): The number of samples to process in a single chunk.
        max_workers (int): The maximum number of parallel workers to use. Defaults
            to -1, which uses all available cores.

    Returns:
        Dataset: The new dataset with language annotations.
    """
    base_dataset.set_format("torch")
    chunk_indices = list(range(0, len(base_dataset), chunk_size))

    with tempfile.TemporaryDirectory() as temp_dir:
        process_chunk_partial = partial(
            process_chunk,
            chunk_size=chunk_size,
            base_dataset=base_dataset,
            scene_vec_desc=scene_vec_desc,
            scene_type=scene_type,
            language_mode=language_mode,
            num_spatial_relation_annotations=num_spatial_relation_annotations,
            subset_probability=subset_probability,
            max_spatial_relationships=max_spatial_relationships,
            spatial_relation_distance_threshold=spatial_relation_distance_threshold,
            temp_dir=temp_dir,
        )

        # Process chunks in parallel.
        chunk_files = process_map(
            process_chunk_partial,
            chunk_indices,
            max_workers=max_workers,
            desc="Processing chunks",
        )

        # Load all chunks and concatenate them into a final dataset.
        chunk_datasets = [
            Dataset.load_from_disk(file, keep_in_memory=False)
            for file in tqdm(chunk_files, desc="Loading chunk datasets")
        ]
        logging.info("Concatenating chunk datasets...")
        final_dataset = concatenate_datasets(chunk_datasets)

    return final_dataset


def add_language_annotation_to_scene(
    item: dict,
    idx: int,
    scene_vec_desc: SceneVecDescription,
    scene_type: SceneType,
    language_mode: LanguageMode,
    num_spatial_relation_annotations: int,
    subset_probability: float,
    max_spatial_relationships: int,
    spatial_relation_distance_threshold: float,
) -> dict:
    scene = item["scenes"]  # Shape (N, V)
    if language_mode == LanguageMode.ALL:
        # Choose based on the index to ensure even distribution of annotation types.
        total = num_spatial_relation_annotations + 2
        language_mode_idx = idx % total

        if language_mode_idx == 0:
            current_language_mode = LanguageMode.OBJECT_NUMBER
        elif language_mode_idx == 1:
            current_language_mode = LanguageMode.OBJECT_NAMES
        else:
            current_language_mode = LanguageMode.SPATIAL_RELATIONS
    else:
        current_language_mode = language_mode

    language_annotation = get_language_annotation(
        scene_vec_desc=scene_vec_desc,
        scene=scene,
        scene_type=scene_type,
        language_mode=current_language_mode,
        num_spatial_relation_annotations=num_spatial_relation_annotations,
        subset_probability=subset_probability,
        max_spatial_relationships=max_spatial_relationships,
        spatial_relation_distance_threshold=spatial_relation_distance_threshold,
    )[0]
    item["language_annotation"] = language_annotation
    return item


def main():
    parser = argparse.ArgumentParser(
        description="Add language annotations to a HF dataset."
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the dataset to add language annotations to.",
    )
    parser.add_argument(
        "annotated_dataset_path",
        type=str,
        help="Path to the new annotated dataset.",
    )
    parser.add_argument(
        "--scene_type",
        type=SceneType,
        choices=list(SceneType),
        required=True,
        help="The scene type that is contained in the dataset.",
    )
    parser.add_argument(
        "--language_mode",
        type=LanguageMode,
        choices=list(LanguageMode),
        required=True,
        help="Language mode to use for annotations.",
    )
    parser.add_argument(
        "--num_spatial_relation_annotations",
        type=int,
        default=1,
        help="Number of spatial relation annotations to include per scene if the "
        "language mode is 'spatial_relations'.",
    )
    parser.add_argument(
        "--subset_probability",
        type=float,
        default=0.5,
        help="The probability that the object name annotation will list a subset of the "
        "objects in the scene.",
    )
    parser.add_argument(
        "--max_spatial_relationships",
        type=int,
        default=5,
        help="The maximum number of spatial relationships to include in the annotation.",
    )
    parser.add_argument(
        "--spatial_relation_distance_threshold",
        type=float,
        default=0.2,
        help="The maximum distance between objects to consider for spatial relationships.",
    )
    parser.add_argument(
        "--do_not_expand_dataset",
        action="store_true",
        help="If true, the dataset will not be expanded and a single annotation will be "
        "generated per scene.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=os.cpu_count(),
        help="The maximum number of parallel workers to use.",
    )
    args = parser.parse_args()
    dataset_path = args.dataset_path
    annotated_dataset_path = args.annotated_dataset_path
    scene_type = args.scene_type
    language_mode = args.language_mode
    num_spatial_relation_annotations = args.num_spatial_relation_annotations
    subset_probability = args.subset_probability
    max_spatial_relationships = args.max_spatial_relationships
    spatial_relation_distance_threshold = args.spatial_relation_distance_threshold
    do_not_expand_dataset = args.do_not_expand_dataset
    max_workers = args.max_workers

    # Set seed for reproducibility.
    random.seed(42)
    torch.manual_seed(42)

    base_dataset, metadata = load_hf_dataset_with_metadata(dataset_path)

    # Load the normalizer.
    normalizer = MinMaxScaler(output_min=-1.0, output_max=1.0, clip=True)
    normalizer.load_serializable_state(metadata["normalizer_state"])
    if not normalizer.is_fitted:
        raise ValueError("Normalizer is not fitted!")

    # Unormalize the scenes.
    unnormalized_dataset = unnormalize_all_scenes(
        normalizer=normalizer, hf_dataset=base_dataset, num_procs=max_workers
    )

    # Create the scene vector description.
    scene_vec_desc = get_scene_vec_description_from_metadata(metadata)

    # Add language annotations to the dataset.
    if do_not_expand_dataset:
        annotated_unnormalized_dataset = unnormalized_dataset.map(
            partial(
                add_language_annotation_to_scene,
                scene_vec_desc=scene_vec_desc,
                scene_type=scene_type,
                language_mode=language_mode,
                num_spatial_relation_annotations=num_spatial_relation_annotations,
                subset_probability=subset_probability,
                max_spatial_relationships=max_spatial_relationships,
                spatial_relation_distance_threshold=spatial_relation_distance_threshold,
            ),
            with_indices=True,
            num_proc=max_workers,
        )
    else:
        annotated_unnormalized_dataset = create_new_dataset_with_language_annotations(
            base_dataset=unnormalized_dataset,
            scene_vec_desc=scene_vec_desc,
            scene_type=scene_type,
            language_mode=language_mode,
            num_spatial_relation_annotations=num_spatial_relation_annotations,
            subset_probability=subset_probability,
            max_spatial_relationships=max_spatial_relationships,
            spatial_relation_distance_threshold=spatial_relation_distance_threshold,
            max_workers=max_workers,
        )

    # For some reason, we need to save and re-load the dataset for parallelism to work
    # again.
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the unnormalized dataset to disk.
        intermediate_path = os.path.join(temp_dir, "tmp_dataset")
        Dataset.save_to_disk(annotated_unnormalized_dataset, intermediate_path)

        # Load the intermediate dataset from the temporary directory.
        tmp_dataset = Dataset.load_from_disk(intermediate_path, keep_in_memory=False)

        # Normalize the scenes.
        annotated_dataset = normalize_all_scenes(
            normalizer=normalizer, hf_dataset=tmp_dataset, num_procs=max_workers
        )

    # Save the new dataset.
    # Using multiple processes can cause issues.
    save_hf_dataset_with_metadata(
        hf_dataset=annotated_dataset,
        metadata=metadata,
        dataset_path=annotated_dataset_path,
        num_procs=1,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
