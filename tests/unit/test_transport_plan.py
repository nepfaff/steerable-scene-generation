import unittest

from functools import cache

import torch

from steerable_scene_generation.algorithms.scene_diffusion.scene_distance import (
    arrange_objects_to_match_scenes,
)


class TestTransportPlan(unittest.TestCase):
    @cache
    def get_scenes(self):
        scene_a = (
            torch.tensor([[[1.0, 2.0, 3.0], [1.0, 3.0, 2.0], [4.0, 4.0, 2.0]]]) / 4
        )
        scene_b = (
            torch.tensor([[[4.0, 4.0, 2.0], [1.0, 3.0, 2.0], [1.0, 2.0, 3.0]]]) / 4
        )
        return scene_a, scene_b

    def test_transport_plan(self):
        scene_a, scene_b = self.get_scenes()

        scene_b_arranged = arrange_objects_to_match_scenes(
            scene_a, scene_b, mask=torch.tensor([True, True, True])
        )
        self.assertTrue(torch.all(scene_b_arranged == scene_a))

    def test_transport_plan_with_masking(self):
        scene_a, scene_b = self.get_scenes()

        scene_b_arranged = arrange_objects_to_match_scenes(
            scene_a, scene_b, mask=torch.tensor([False, True, False])
        )
        self.assertTrue(torch.all(scene_b_arranged == scene_a))

    def test_ambiguous(self):
        # Two equally likely matches.
        scene_a = (
            torch.tensor([[[1.0, 2.0, 3.0], [1.0, 3.0, 2.0], [1.0, 2.0, 3.0]]]) / 4
        )
        scene_b = (
            torch.tensor([[[1.0, 3.0, 2.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]]) / 4
        )

        scene_b_arranged = arrange_objects_to_match_scenes(
            scene_a, scene_b, mask=torch.tensor([True, True, True])
        )
        self.assertTrue(torch.all(scene_b_arranged == scene_a))


if __name__ == "__main__":
    unittest.main()
