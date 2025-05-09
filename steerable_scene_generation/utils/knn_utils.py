import torch

from tqdm import tqdm

from steerable_scene_generation.algorithms.scene_diffusion.scene_distance import (
    compute_distances_to_training_examples,
)
from steerable_scene_generation.datasets import SceneDataset


def get_k_closest_training_examples(
    scenes: torch.Tensor, dataset: SceneDataset, num_k: int, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the k closest training examples for each scene in a batch according to the
    Sinkhorn-Knopp distance between the object permuation equivariant scene vectors.

    Args:
        scenes (torch.Tensor): The unormalized scenes to find the closest training
            examples for. Shape (B, N, V).
        dataset (TableSceneDataset): The dataset containing normalized training scenes.
        num_k (int): The number of closest training examples to return.
        batch_size (int): The batch size to use when processing the dataset.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple of (closest_scenes, distances) where:
            - closest_scenes: Tensor of shape (B, K, N, V) containing the k closest
              training examples for each scene in the batch.
            - distances: Tensor of shape (B, K) containing the distances to the k
              closest training examples for each scene in the batch.
    """

    # Normalize the input scenes.
    scenes_normalized = dataset.normalize_scenes(scenes)  # Shape (B, N, V)

    # Initialize arrays to track the k closest examples for each scene.
    B = scenes.shape[0]
    device = scenes.device
    top_k_distances = torch.full((B, num_k), float("inf"), device=device)
    top_k_indices = torch.zeros((B, num_k), dtype=torch.long, device=device)

    # Process the dataset in batches.
    dataset_size = len(dataset)
    for start_idx in tqdm(
        range(0, dataset_size, batch_size), desc="Getting k-closest training examples"
    ):
        end_idx = min(start_idx + batch_size, dataset_size)

        # Get a batch of training scenes.
        batch_indices = torch.arange(start_idx, end_idx)
        batch_data = dataset.get_all_data(
            normalized=True, scene_indices=batch_indices, only_scenes=True
        )
        training_batch = batch_data["scenes"].to(device)  # Shape (batch_size, N, V)

        # Compute distances between input scenes and this batch of training scenes.
        batch_distances = compute_distances_to_training_examples(
            scenes_normalized, training_batch
        )  # Shape (B, batch_size)

        # For each scene, update the top-k closest examples if we found better ones.
        for i in range(B):
            # Combine current top-k with new distances.
            combined_distances = torch.cat([top_k_distances[i], batch_distances[i]])
            combined_indices = torch.cat(
                [top_k_indices[i], torch.arange(start_idx, end_idx, device=device)]
            )

            # Get the top-k from the combined set.
            if len(combined_distances) > num_k:
                _, sorted_indices = torch.topk(
                    combined_distances, k=num_k, largest=False
                )
                top_k_distances[i] = combined_distances[sorted_indices]
                top_k_indices[i] = combined_indices[sorted_indices]

    # Create tensors to store the results.
    closest_scenes_tensor = torch.zeros(
        (B, num_k, scenes.shape[1], scenes.shape[2]), device=scenes.device
    )
    distances_tensor = torch.zeros((B, num_k), device=scenes.device)

    # Retrieve the actual scenes for the top-k indices.
    for i in range(B):
        scene_indices = top_k_indices[i].cpu()
        # Get the scenes from the dataset and inverse normalize them.
        scene_data = dataset.get_all_data(normalized=False, scene_indices=scene_indices)
        closest_scenes = scene_data["scenes"].to(scenes.device)

        # Store in the result tensors.
        closest_scenes_tensor[i] = closest_scenes
        distances_tensor[i] = top_k_distances[i]

    return closest_scenes_tensor, distances_tensor
