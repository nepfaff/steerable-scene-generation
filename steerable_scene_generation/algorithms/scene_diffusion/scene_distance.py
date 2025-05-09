from typing import Callable, Tuple, Union

import torch
import torch.nn as nn


class SinkhornKnoppSolver(nn.Module):
    """
    Implements the Sinkhorn-Knopp algorithm to compute the optimal transport plan
    between two empirical measures.

    Uses the log-domain to increase numerical stability.

    Adapted from https://gist.github.com/wohlert/8589045ab544082560cc5f8915cc90bd.
    """

    def __init__(
        self,
        epsilon: float = 1e-2,
        iterations: int = 100,
        early_stop_threshold: float = 1e-4,
        metric: Callable[[torch.Tensor], torch.Tensor] = lambda x: torch.pow(x, 2),
        return_transport_plan: bool = False,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.iterations = iterations
        self.early_stop_threshold = early_stop_threshold
        self.metric = metric
        self.return_transport_plan = return_transport_plan

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        num_x = x.size(-2)
        num_y = y.size(-2)

        batch_size = x.size(0) if x.dim() > 2 else 1

        # Marginal densities are empirical measures.
        a = x.new_ones((batch_size, num_x), requires_grad=False) / num_x
        b = y.new_ones((batch_size, num_y), requires_grad=False) / num_y

        # Initialise approximation vectors in log domain.
        u = torch.zeros_like(a)
        v = torch.zeros_like(b)

        # Cost matrix.
        C = self._compute_cost(x, y)

        # Sinkhorn iterations.
        for _ in range(self.iterations):
            u0, v0 = u, v

            # u^{l+1} = a / (K v^l)
            K = self._get_boltzmann_kernel(u, v, C)
            u_ = torch.log(a + 1e-8) - torch.logsumexp(K, dim=-1)
            u = self.epsilon * u_ + u

            # v^{l+1} = b / (K^T u^(l+1))
            K_t = self._get_boltzmann_kernel(u, v, C).transpose(-2, -1)
            v_ = torch.log(b + 1e-8) - torch.logsumexp(K_t, dim=-1)
            v = self.epsilon * v_ + v

            # Size of the change we have performed on u.
            diff = torch.sum(torch.abs(u - u0), dim=-1) + torch.sum(
                torch.abs(v - v0), dim=-1
            )
            mean_diff = torch.mean(diff)
            if mean_diff.item() < self.early_stop_threshold:
                break

        # Transport plan pi = diag(a)*K*diag(b).
        K = self._get_boltzmann_kernel(u, v, C)
        pi = torch.exp(K)

        # Sinkhorn distance.
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.return_transport_plan:
            return cost, pi
        return cost

    def _compute_cost(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_ = x.unsqueeze(-2)
        y_ = y.unsqueeze(-3)
        C = torch.sum(self.metric(x_ - y_), dim=-1)
        return C

    def _get_boltzmann_kernel(
        self, u: torch.Tensor, v: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor:
        kernel = -C + u.unsqueeze(-1) + v.unsqueeze(-2)
        kernel /= self.epsilon
        return kernel


def compute_transport_plan(
    scene1: torch.Tensor,
    scene2: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float = 1e-2,
    iterations: int = 300,
    early_stop_threshold: float = 1e-4,
) -> torch.Tensor:
    """
    Compute the Sinkhorn-Knopp transport plan between two scenes. The transport plan
    is a (N, N) matrix where N is the number of objects in the scene. The transport plan
    (i, j) element represents the amount of mass transported from object i in scene1 to
    object j in scene2. One can threshold the transport plan to obtain a matching
    between the objects in the two scenes.

    Args:
        scene1 (torch.Tensor): Scene of shape (B, N, V) where N are the number of
            objects and V is the object feature vector length.
        scene2 (torch.Tensor): Scene of shape (B, N, V) where N are the number of
            objects and V is the object feature vector length.
        mask (torch.Tensor): Boolean mask of shape (V,) where V is the object feature
            vector length. The mask is used to select the object features to compute the
            transport plan.
        epsilon (float, optional): Regularization parameter. Defaults to 1e-2.
        iterations (int, optional): Number of Sinkhorn iterations. Defaults to 300.
        early_stop_threshold (float, optional): Early stopping threshold. Defaults to
            1e-4.

    Returns:
        torch.Tensor: Transport plan between the scenes. Shape (B, N, N).
    """
    assert scene1.dim() == 3
    assert scene1.size() == scene2.size()
    assert mask.dim() == 1 and mask.size(0) == scene1.size(-1)

    sinkhorn_solver = SinkhornKnoppSolver(
        epsilon=epsilon,
        iterations=iterations,
        early_stop_threshold=early_stop_threshold,
        return_transport_plan=True,
    )
    mask = mask.unsqueeze(0).unsqueeze(0).expand(scene1.size(0), scene1.size(1), -1)
    scene1_masked = scene1 * mask
    scene2_masked = scene2 * mask
    _, transport_plan = sinkhorn_solver(scene1_masked, scene2_masked)
    return transport_plan


def arrange_objects_to_match_scenes(
    scene1: torch.Tensor,
    scene2: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float = 1e-2,
    iterations: int = 300,
    early_stop_threshold: float = 1e-4,
    transport_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Arranges the objects in `scene2` to match the objects in `scene1` using the
    Sinkhorn-Knopp transport plan. The transport plan is computed using the object
    features selected by the `mask`. The first scene in the `scene1` batch is matched
    with the first scene in the `scene2` batch, and so on.

    Args:
        scene1 (torch.Tensor): Scene of shape (B, N, V) where N are the number of
            objects and V is the object feature vector length.
        scene2 (torch.Tensor): Scene of shape (B, N, V) where N are the number of
            objects and V is the object feature vector length.
        mask (torch.Tensor): Boolean mask of shape (V,) where V is the object feature
            vector length. The mask is used to select the object features to compute the
            transport plan.
        epsilon (float, optional): Regularization parameter. Defaults to 1e-2.
        iterations (int, optional): Number of Sinkhorn iterations. Defaults to 300.
        early_stop_threshold (float, optional): Early stopping threshold. Defaults to
            1e-4.
        transport_threshold (float, optional): Threshold to obtain unique matching.
            The transport plan is normalized to [0, 1] before thresholding. Defaults to
            0.5.

    Returns:
        torch.Tensor: Scene2 with objects arranged to match the objects in scene1.
            Shape (B, N, V).
    """
    transport_plan = compute_transport_plan(
        scene1, scene2, mask, epsilon, iterations, early_stop_threshold
    )

    # Normalize the transport plan to the range [0, 1]
    min_val = torch.min(transport_plan, dim=-1, keepdim=True)[0]
    max_val = torch.max(transport_plan, dim=-1, keepdim=True)[0]
    transport_plan_normalized = (transport_plan - min_val) / (max_val - min_val)

    transport_plan_thresholded = (
        transport_plan_normalized > transport_threshold
    ).float()

    # Ensure uniqueness by resolving ambiguities randomly.
    batch_size, num_objects, _ = transport_plan_thresholded.shape
    for b in range(batch_size):
        for i in range(num_objects):
            if torch.sum(transport_plan_thresholded[b, i]) > 1:
                # Get indices of non-zero elements.
                non_zero_indices = torch.nonzero(
                    transport_plan_thresholded[b, i]
                ).squeeze()
                # Randomly select one index.
                random_index = non_zero_indices[
                    torch.randint(len(non_zero_indices), (1,))
                ]
                # Set all other elements to zero.
                transport_plan_thresholded[b, i] = torch.zeros_like(
                    transport_plan_thresholded[b, i]
                )
                transport_plan_thresholded[b, i, random_index] = 1.0

    scene2_matched = torch.bmm(transport_plan_thresholded, scene2)
    return scene2_matched


def compute_scene_distance(
    scene1: torch.Tensor,
    scene2: torch.Tensor,
    epsilon: float = 1e-2,
    iterations: int = 100,
    early_stop_threshold: float = 1e-4,
) -> torch.Tensor:
    """
    Compute the Sinkhorn-Knopp distance between two scenes which can be used as a
    distance measure between two unordered sets of objects. The algorithm runs in O(N^3)
    time where N is the number of objects in the scene. It uses the squared Euclidean
    distance as the metric.

    Note that the distance is more intuitive when the scene tensors are normalized.

    Args:
        scene1 (torch.Tensor): Scene of shape (B, N, V) where N are the number of
            objects and V is the object feature vector length.
        scene2 (torch.Tensor): Scene of shape (B, N, V) where N are the number of
            objects and V is the object feature vector length.
        epsilon (float, optional): Regularization parameter. Defaults to 1e-2.
        iterations (int, optional): Number of Sinkhorn iterations. Defaults to 100.
        early_stop_threshold (float, optional): Early stopping threshold. Defaults to
            1e-4.

    Returns:
        torch.Tensor: Sinkhorn-Knopp distance between the scenes of shape (B,).
    """
    sinkhorn_solver = SinkhornKnoppSolver(
        epsilon=epsilon,
        iterations=iterations,
        early_stop_threshold=early_stop_threshold,
    ).to(scene1.device)
    return sinkhorn_solver(scene1, scene2)


def compute_distances_to_training_examples(
    scenes: torch.Tensor, training_scenes: torch.Tensor
) -> torch.Tensor:
    """
    Compute the distances between a scene and all training examples.

    Args:
        scenes (torch.Tensor): Scenes of shape (B, N, V) where N are the number of
            objects and V is the object feature vector length.
        training_scenes (torch.Tensor): Training scenes of shape (M, N, V) where M
            is the number of training examples, N are the number of objects and V is
            the object feature vector length.

    Returns:
        torch.Tensor: The distances between the scene and the training examples for each
            scene in the batch. Shape (B, M).
    """
    assert scenes.dim() == 3

    B, N, V = scenes.shape
    M = training_scenes.shape[0]

    # Expand dimensions to handle batch processing.
    scenes_expanded = scenes.unsqueeze(1).expand(-1, M, -1, -1)  # Shape (B, M, N, V)
    training_scenes_expanded = training_scenes.unsqueeze(0).expand(
        B, -1, -1, -1
    )  # Shape (B, M, N, V)

    # Flatten the expanded tensors for batch distance computation.
    scenes_flattened = scenes_expanded.reshape(-1, N, V)  # Shape (B*M, N, V)
    training_scenes_flattened = training_scenes_expanded.reshape(
        -1, N, V
    )  # Shape (B*M, N, V)

    # Compute the pairwise distances in a batched manner.
    distances = compute_scene_distance(
        scenes_flattened, training_scenes_flattened
    )  # Shape (B*M,)
    distances = distances.reshape(B, M)  # Shape (B, M)

    return distances


def compute_distance_to_closest_training_example(
    scenes: torch.Tensor, training_scenes: torch.Tensor
) -> torch.Tensor:
    """
    Compute the distance between a scene and the closest training example.

    Args:
        scenes (torch.Tensor): Scenes of shape (B, N, V) where N are the number of
            objects and V is the object feature vector length.
        training_scenes (torch.Tensor): Training scenes of shape (M, N, V) where M
            is the number of training examples, N are the number of objects and V is
            the object feature vector length.

    Returns:
        torch.Tensor: The distance between the scene and the closest training example
            for each scene in the batch. Shape (B,).
    """
    distances = compute_distances_to_training_examples(
        scenes, training_scenes
    )  # Shape (B, M)
    return torch.min(distances, dim=1).values  # Shape (B,)


def compute_all_pairwise_scene_distances(
    scenes: torch.Tensor, batch_size: int = 100
) -> torch.Tensor:
    """
    Compute all pairwise distances between scenes.

    Args:
        scenes (torch.Tensor): Scenes of shape (B, N, V) where N are the number of
            objects and V is the object feature vector length.
        batch_size (int): The number of scenes to process in each batch. Default is 100.

    Returns:
        torch.Tensor: The distances between all pairs of scenes. Shape (B, B).
    """
    B, N, V = scenes.shape
    all_distances = torch.full((B, B), float("inf"), device=scenes.device)

    for i in range(0, B, batch_size):
        end_i = min(i + batch_size, B)
        scenes_batch_1 = scenes[i:end_i]

        for j in range(0, B, batch_size):
            end_j = min(j + batch_size, B)
            scenes_batch_2 = scenes[j:end_j]

            # Expand dimensions to handle batch processing.
            scenes_expanded_1 = scenes_batch_1.unsqueeze(1).expand(
                -1, end_j - j, -1, -1
            )
            scenes_expanded_2 = scenes_batch_2.unsqueeze(0).expand(
                end_i - i, -1, -1, -1
            )

            # Flatten the expanded tensors for batch distance computation.
            scenes_flattened_1 = scenes_expanded_1.reshape(-1, N, V)
            scenes_flattened_2 = scenes_expanded_2.reshape(-1, N, V)

            # Compute the pairwise distances in a batched manner.
            distances = compute_scene_distance(
                scenes_flattened_1, scenes_flattened_2
            ).reshape(end_i - i, end_j - j)

            # Store the computed distances in the all_distances tensor.
            all_distances[i:end_i, j:end_j] = distances

    return all_distances


def compute_min_scene_distances(
    scenes: torch.Tensor, batch_size: int = 100, return_pairwise_distances: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute the minimum distance between a scene and all other scenes.

    Args:
        scenes (torch.Tensor): Scenes of shape (B, N, V) where N are the number of
            objects and V is the object feature vector length.
        batch_size (int): The number of scenes to process in each batch. Default is 100.
        return_pairwise_distances (bool): Whether also to return the pairwise distances.

    Returns:
        One of the following:
            - torch.Tensor: The minimum distance between the scene and all other scenes
                for each scene in the batch. Shape (B,).
            - Tuple[torch.Tensor, torch.Tensor]: A tuple of the minimum distances of
                shape (B,) and the pairwise distances of shape (B, B).
    """
    # Compute all pairwise distances using the helper function.
    pairwise_distances = compute_all_pairwise_scene_distances(
        scenes, batch_size=batch_size
    )

    # Mask self-comparisons (diagonal elements).
    mask = torch.eye(
        pairwise_distances.shape[0], dtype=torch.bool, device=pairwise_distances.device
    )
    pairwise_distances.masked_fill_(mask, float("inf"))

    # Compute the minimum distance to all other scenes for each scene.
    min_distances = pairwise_distances.min(dim=1).values

    if return_pairwise_distances:
        return min_distances, pairwise_distances
    return min_distances


def compute_mean_min_scene_distance(
    scenes: torch.Tensor, num_workers: int = 1, use_pbar: bool = True
) -> torch.Tensor:
    """
    Compute the mean minimum distance between a scene and all other scenes.

    Args:
        scenes (torch.Tensor): Scenes of shape (B, N, V) where N are the number of
            objects and V is the object feature vector length.
        num_workers (int, optional): Number of workers to use for parallel processing.
            Note that using more than one worker is likely slower until the number of
            scenes becomes huge.
        use_pbar (bool, optional): Whether to use a progress bar. Defaults to True.

    Returns:
        torch.Tensor: The mean minimum distance between the scene and all other scenes.
            Shape (1,).
    """
    distances = compute_min_scene_distances(scenes, num_workers, use_pbar)  # Shape (B,)
    return torch.mean(distances)
