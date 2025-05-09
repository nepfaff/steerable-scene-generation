import argparse
import csv
import os

import matplotlib.pyplot as plt

from steerable_scene_generation.utils.hf_dataset import (
    get_scene_vec_description_from_metadata,
    load_hf_dataset_with_metadata,
    unnormalize_all_scenes,
)
from steerable_scene_generation.utils.min_max_scaler import MinMaxScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the HF dataset to plot the histogram for.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save the histogram plot.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=os.cpu_count(),
        help="The maximum number of parallel workers to use.",
    )
    args = parser.parse_args()
    dataset_path = args.dataset_path
    output_dir = args.output_dir
    max_workers = args.max_workers

    dataset, metadata = load_hf_dataset_with_metadata(dataset_path)
    scene_vec_desc = get_scene_vec_description_from_metadata(metadata)

    # Load the normalizer.
    normalizer = MinMaxScaler(output_min=-1.0, output_max=1.0, clip=True)
    normalizer.load_serializable_state(metadata["normalizer_state"])
    if not normalizer.is_fitted:
        raise ValueError("Normalizer is not fitted!")

    # Unormalize the scenes.
    unnormalized_dataset = unnormalize_all_scenes(
        normalizer=normalizer, hf_dataset=dataset, num_procs=max_workers
    )
    unnormalized_dataset.set_format("torch")

    # Count the number of objects in each scene.
    def count_objects(item):
        scene = item["scenes"]  # Shape (N, V)
        num_objects = sum(
            1 for obj in scene if scene_vec_desc.get_model_path(obj) is not None
        )
        return {"num_objects": num_objects}

    object_counts = unnormalized_dataset.map(
        count_objects,
        num_proc=max_workers,
        batched=False,
        desc="Counting objects",
        remove_columns=unnormalized_dataset.column_names,
    )

    # Convert to a list and ensure integers.
    num_objects_list = [int(count) for count in object_counts["num_objects"]]

    # Plot the histogram.
    plt.figure(figsize=(10, 6))
    min_objects = min(num_objects_list)
    max_objects = max(num_objects_list)
    plt.hist(
        num_objects_list,
        bins=range(min_objects, max_objects + 2),  # Align bins to integer values
        color="blue",
        alpha=0.7,
        edgecolor="black",
    )
    plt.title("Distribution of Number of Objects in Scenes")
    plt.xlabel("Number of Objects")
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save the plot.
    output_path = os.path.join(output_dir, "object_count_histogram.png")
    plt.savefig(output_path)
    print(f"Histogram saved to {output_path}")

    # Count frequencies of each object number.
    object_frequencies = {}
    for count in num_objects_list:
        object_frequencies[count] = object_frequencies.get(count, 0) + 1

    # Save the histogram data.
    output_path = os.path.join(output_dir, "object_count_histogram.csv")
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["num_objects", "frequency"])  # header
        # Write rows from min to max objects, including zeros.
        for num_obj in range(min_objects, max_objects + 1):
            writer.writerow([num_obj, object_frequencies.get(num_obj, 0)])
    print(f"Histogram data saved to {output_path}")


if __name__ == "__main__":
    main()
