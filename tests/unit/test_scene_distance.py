import random
import unittest

from itertools import permutations

import torch

from steerable_scene_generation.algorithms.scene_diffusion.scene_distance import (
    compute_all_pairwise_scene_distances,
    compute_distances_to_training_examples,
    compute_min_scene_distances,
    compute_scene_distance,
)


def gt_mse_scene_distance(scene_a: torch.Tensor, scene_b: torch.Tensor) -> torch.Tensor:
    """
    Computes the minimum MSE distance between two unordered scenes by considering all
    permutations. This is a brute force method that runs in O(n!) time where n is the
    number of objects in the scene.
    """
    min_distance = float("inf")
    for perm in permutations(scene_b):
        distance = torch.sum(
            torch.tensor(
                [torch.mean((scene_a[i] - perm[i]) ** 2) for i in range(len(scene_a))]
            )
        )
        if distance < min_distance:
            min_distance = distance

    return min_distance


class TestSceneDistance(unittest.TestCase):
    def test_scene_distance(self):
        scene_a = torch.tensor([[1.0, 2.0, 3.0], [1.0, 3.0, 2.0], [4.0, 4.0, 2.0]]) / 4
        scene_b = torch.tensor([[4.0, 4.0, 2.0], [2.0, 3.0, 2.0], [1.0, 2.0, 3.0]]) / 4
        scene_c = torch.tensor([[1.0, 2.0, 2.0], [1.0, 4.0, 2.0], [4.0, 1.0, 4.0]]) / 4
        scene_d = torch.tensor([[1.0, 3.0, 2.0], [1.0, 2.0, 3.0], [4.0, 4.0, 2.0]]) / 4
        scene_e = torch.tensor([[1.0, 4.0, 2.0], [1.0, 2.0, 2.0], [4.0, 1.0, 4.0]]) / 4
        scene_f = (
            torch.tensor([[-1.0, 2.0, 3.0], [1.1, 3.0, 2.5], [4.0, 4.0, -3.2]]) / 4
        )

        # Test that the distance is permutation invariant.
        self.assertAlmostEqual(
            compute_scene_distance(scene_a, scene_c).item(),
            compute_scene_distance(scene_a, scene_e).item(),
            places=2,
        )
        self.assertAlmostEqual(
            compute_scene_distance(scene_c, scene_e).item(), 0.0, places=2
        )

        # Test all scene combinations.
        scenes = [scene_a, scene_b, scene_c, scene_d, scene_e, scene_f]
        for i in range(len(scenes)):
            for j in range(len(scenes)):
                distance_ij = compute_scene_distance(scenes[i], scenes[j]).item()
                distance_ij_gt = gt_mse_scene_distance(scenes[i], scenes[j]).item()
                self.assertAlmostEqual(distance_ij, distance_ij_gt, places=2)

        # Test batched.
        scenes1 = torch.stack(scenes)
        random.shuffle(scenes)
        scenes2 = torch.stack(scenes)
        distances = compute_scene_distance(scenes1, scenes2)
        for i in range(len(scenes)):
            distance_gt = gt_mse_scene_distance(scenes1[i], scenes2[i]).item()
            self.assertAlmostEqual(distances[i].item(), distance_gt, places=2)

    def test_compute_distances_to_training_examples(self):
        # Example scenes with shape (B, N, V).
        scenes = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])

        # Example training scenes with shape (M, N, V).
        training_scenes = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[2.0, 1.0], [3.0, 2.0]],
                [[4.0, 3.0], [5.0, 4.0]],
            ]
        )

        # Compute expected distances.
        B = scenes.shape[0]
        M = training_scenes.shape[0]
        expected_distances = torch.zeros(B, M)
        for i in range(B):
            for j in range(M):
                expected_distances[i, j] = gt_mse_scene_distance(
                    scenes[i], training_scenes[j]
                )

        # Compute distances using the function.
        distances = compute_distances_to_training_examples(scenes, training_scenes)

        self.assertTrue(torch.allclose(distances, expected_distances, rtol=1e-2))

    def test_compute_all_pairwise_scene_distances(self):
        # Example scenes with shape (B, N, V).
        scenes = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ]
        )

        # Compute expected pairwise distances.
        B = scenes.shape[0]
        expected_distances = torch.zeros(B, B)
        for i in range(B):
            for j in range(B):
                expected_distances[i, j] = gt_mse_scene_distance(scenes[i], scenes[j])

        # Compute pairwise distances using the function.
        all_distances = compute_all_pairwise_scene_distances(scenes)

        self.assertTrue(torch.allclose(all_distances, expected_distances, rtol=1e-2))

    def test_compute_min_scene_distances(self):
        # Example scenes with shape (B, N, V).
        scenes = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ]
        )

        # Compute expected minimum distances.
        B = scenes.shape[0]
        expected_min_distances = torch.zeros(B)
        for i in range(B):
            min_distance = float("inf")
            for j in range(B):
                if i != j:
                    distance = gt_mse_scene_distance(scenes[i], scenes[j])
                    if distance < min_distance:
                        min_distance = distance
            expected_min_distances[i] = min_distance

        # Compute minimum distances using the function.
        min_distances = compute_min_scene_distances(scenes)
        self.assertTrue(
            torch.allclose(min_distances, expected_min_distances, rtol=1e-2)
        )

        # Compute in batches.
        min_distances_batches = compute_min_scene_distances(scenes, batch_size=2)
        self.assertTrue(
            torch.allclose(min_distances_batches, expected_min_distances, rtol=1e-2)
        )


if __name__ == "__main__":
    unittest.main()
