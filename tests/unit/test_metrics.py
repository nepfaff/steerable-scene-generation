import unittest

import torch

from pydrake.all import PackageMap

from steerable_scene_generation.algorithms.common.dataclasses import (
    RotationParametrization,
    SceneVecDescription,
)
from steerable_scene_generation.utils.prompt_following_metrics import (
    compute_prompt_following_metrics,
)
from steerable_scene_generation.utils.scene_metrics import (
    compute_scene_object_kl_divergence_metric,
    compute_total_minimum_distance,
    compute_total_scene_penetration,
    compute_welded_object_pose_deviation_metric,
)


class TestSceneMetrics(unittest.TestCase):
    def test_compute_total_scene_penetration(self):
        scene_vec_desc = SceneVecDescription(
            drake_package_map=PackageMap(),
            static_directive=None,
            translation_vec_len=3,
            rotation_parametrization=RotationParametrization.AXIS_ANGLE,
            model_paths=["tests/models/box.sdf"] * 2,
            model_path_vec_len=None,
        )

        # Test zero penetration.
        scene = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Box1
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Box2
            ]
        )
        scene_penetration = compute_total_scene_penetration(
            scene=scene, scene_vec_desc=scene_vec_desc
        )
        self.assertTrue(scene_penetration == 0.0)

        # Test non-zero penetration.
        scene = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Box1
                [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],  # Box2
            ]
        )
        scene_penetration = compute_total_scene_penetration(
            scene=scene, scene_vec_desc=scene_vec_desc
        )
        self.assertTrue(scene_penetration == 0.5)

    def test_compute_total_minimum_distance(self):
        scene_vec_desc = SceneVecDescription(
            drake_package_map=PackageMap(),
            static_directive=None,
            translation_vec_len=3,
            rotation_parametrization=RotationParametrization.AXIS_ANGLE,
            model_paths=["tests/models/box.sdf"] * 2,
            model_path_vec_len=None,
        )

        # Test zero distance.
        scene = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Box1
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Box2
            ]
        )
        minimum_distance = compute_total_minimum_distance(
            scene=scene, scene_vec_desc=scene_vec_desc
        )
        self.assertTrue(minimum_distance == 0.0)

        # Test less than zero distance.
        scene = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Box1
                [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],  # Box2
            ]
        )
        minimum_distance = compute_total_minimum_distance(
            scene=scene, scene_vec_desc=scene_vec_desc
        )
        self.assertTrue(minimum_distance == 0.0)

        # Test greater than zero distance.
        scene = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Box1
                [1.5, 0.0, 0.0, 0.0, 0.0, 0.0],  # Box2
            ]
        )
        minimum_distance = compute_total_minimum_distance(
            scene=scene, scene_vec_desc=scene_vec_desc
        )
        # Distance counted twice in this case as none of the objects is static.
        self.assertTrue(minimum_distance == 1.0)

        # Test more than two objects.
        scene_vec_desc = SceneVecDescription(
            drake_package_map=PackageMap(),
            static_directive=None,
            translation_vec_len=3,
            rotation_parametrization=RotationParametrization.AXIS_ANGLE,
            model_paths=["tests/models/box.sdf"] * 5,
            model_path_vec_len=None,
        )
        scene = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Box1
                [1.5, 0.0, 0.0, 0.0, 0.0, 0.0],  # Box2, 0.5m distance to Box1
                [3.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Box3, 0.5m distance to Box2
                [4.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Box4, zero distance to Box3
                [4.5, 0.0, 0.0, 0.0, 0.0, 0.0],  # Box5, zero distance to Box4
            ]
        )
        minimum_distance = compute_total_minimum_distance(
            scene=scene, scene_vec_desc=scene_vec_desc
        )
        self.assertTrue(minimum_distance == 1.0)

    def test_compute_scene_object_kl_divergence_metric(self):
        # Define the SceneVecDescription for both dataset and synthesized scenes.
        scene_vec_desc = SceneVecDescription(
            drake_package_map=PackageMap(),
            static_directive=None,
            translation_vec_len=3,
            rotation_parametrization=RotationParametrization.AXIS_ANGLE,
            model_paths=["tests/models/box.sdf", "tests/models/sphere.sdf"],
            model_path_vec_len=3,  # Box, sphere, and [empty]
        )

        # Dataset scenes (two objects of each type).
        dataset_scenes = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # Box
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # Sphere
                [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # Box
                [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # Sphere
            ]
        )

        # Synthesized scenes (uneven distribution).
        synthesized_scenes = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # Box
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # Sphere
                [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # Sphere
            ]
        )

        # Compute KL divergence.
        kl_divergence = compute_scene_object_kl_divergence_metric(
            dataset_scenes=dataset_scenes,
            dataset_scene_vec_desc=scene_vec_desc,
            synthesized_scenes=synthesized_scenes,
            synthesized_scene_vec_desc=scene_vec_desc,
        )

        # Expected distribution.
        expected_dataset_distribution = torch.tensor(
            [0.5, 0.5]
        )  # Equal counts of Box and Sphere
        expected_synthesized_distribution = torch.tensor(
            [1 / 3, 2 / 3]
        )  # Unequal counts

        # Manually compute expected KL divergence.
        expected_kl = (
            (
                expected_dataset_distribution
                * (
                    torch.log(expected_dataset_distribution + 1e-6)
                    - torch.log(expected_synthesized_distribution + 1e-6)
                )
            )
            .sum()
            .item()
        )

        # Assert the computed KL divergence matches the expected value.
        self.assertAlmostEqual(kl_divergence, expected_kl, places=6)

    def test_compute_prompt_following_metrics(self):
        # Define the SceneVecDescription with test objects.
        scene_vec_desc = SceneVecDescription(
            drake_package_map=PackageMap(),
            static_directive=None,
            translation_vec_len=3,
            rotation_parametrization=RotationParametrization.AXIS_ANGLE,
            model_paths=[
                "models/apple.sdf",
                "models/bowl.sdf",
                "models/plate.sdf",
            ],
            model_path_vec_len=4,  # apple, bowl, plate, and [empty]
        )

        # Test case 1: Object number prompts.
        prompts = [
            "A scene with 2 objects.",
            "A scene with 1 objects.",
        ]
        scenes = torch.tensor(
            [
                # Scene 1: Has 2 objects (apple, bowl).
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # apple
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # bowl
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # empty
                ],
                # Scene 2: Has 2 objects (plate, bowl) but should have 1.
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # plate
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # bowl
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # empty
                ],
            ]
        )

        metrics = compute_prompt_following_metrics(
            scene_vec_desc=scene_vec_desc, prompts=prompts, scenes=scenes
        )

        self.assertEqual(metrics["num_identifiable_prompts"], 2)
        self.assertEqual(metrics["identifiable_prompt_fraction"], 1.0)
        self.assertEqual(metrics["object_number_prompt_following_fraction"], 0.5)
        self.assertEqual(len(metrics["per_prompt_following_fractions"]), 2)
        # First prompt is followed correctly (2 objects requested, 2 objects present).
        self.assertEqual(metrics["per_prompt_following_fractions"][0], 1.0)
        # Second prompt is not followed correctly (1 object requested, 2 objects present).
        self.assertEqual(metrics["per_prompt_following_fractions"][1], 0.0)
        self.assertEqual(metrics["binary_prompt_satisfaction_rate"], 0.5)
        self.assertEqual(metrics["binary_object_number_satisfaction_rate"], 0.5)
        # First prompt: exactly 2 objects as requested (satisfied).
        self.assertEqual(metrics["per_prompt_binary_satisfaction"][0], 1)
        # Second prompt: 2 objects instead of 1 (not satisfied).
        self.assertEqual(metrics["per_prompt_binary_satisfaction"][1], 0)

        # Test case 2: Object name prompts.
        prompts = [
            "A scene with two apples and a bowl.",
            "A scene with a plate.",
        ]
        scenes = torch.tensor(
            [
                # Scene 1: Has 1 apple and 1 bowl (should have 2 apples).
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # apple
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # bowl
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # empty
                ],
                # Scene 2: Has exactly 1 plate as specified.
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # plate
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # empty
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # empty
                ],
            ]
        )

        metrics = compute_prompt_following_metrics(
            scene_vec_desc=scene_vec_desc, prompts=prompts, scenes=scenes
        )

        self.assertEqual(metrics["num_identifiable_prompts"], 2)
        self.assertEqual(metrics["identifiable_prompt_fraction"], 1.0)
        # First scene: 2/3 correct (1 apple + 1 bowl out of 3 total objects).
        # Second scene: 1/1 correct.
        self.assertAlmostEqual(
            metrics["object_name_prompt_following_fraction"], 0.833333, places=5
        )
        self.assertEqual(len(metrics["per_prompt_following_fractions"]), 2)
        # First prompt: 2/3 correct (1 apple instead of 2, 1 bowl as requested).
        self.assertAlmostEqual(
            metrics["per_prompt_following_fractions"][0], 2 / 3, places=5
        )
        # Second prompt: 1/1 correct (1 plate as requested).
        self.assertEqual(metrics["per_prompt_following_fractions"][1], 1.0)
        self.assertEqual(metrics["binary_prompt_satisfaction_rate"], 0.5)
        self.assertEqual(metrics["binary_object_name_satisfaction_rate"], 0.5)
        # First prompt: only 1 apple instead of 2 (not satisfied).
        self.assertEqual(metrics["per_prompt_binary_satisfaction"][0], 0)
        # Second prompt: exactly 1 plate as requested (satisfied).
        self.assertEqual(metrics["per_prompt_binary_satisfaction"][1], 1)

        # Test case 3: Mixed prompts.
        prompts = [
            "A scene with 2 objects.",
            "A scene with two apples.",
            "Not a valid prompt format",
        ]
        scenes = torch.tensor(
            [
                # Scene 1: Has exactly 2 objects.
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # apple
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # bowl
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # empty
                ],
                # Scene 2: Has exactly 2 apples.
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # apple
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # apple
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # empty
                ],
                # Scene 3: Doesn't matter (invalid prompt).
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # apple
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # empty
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # empty
                ],
            ]
        )

        metrics = compute_prompt_following_metrics(
            scene_vec_desc=scene_vec_desc, prompts=prompts, scenes=scenes
        )

        self.assertEqual(metrics["num_identifiable_prompts"], 2)
        self.assertEqual(metrics["identifiable_prompt_fraction"], 2 / 3)
        self.assertEqual(metrics["prompt_following_fraction"], 1.0)
        self.assertEqual(len(metrics["per_prompt_following_fractions"]), 3)
        # First prompt: 2 objects requested, 2 objects present.
        self.assertEqual(metrics["per_prompt_following_fractions"][0], 1.0)
        # Second prompt: 2 apples requested, 2 apples present.
        self.assertEqual(metrics["per_prompt_following_fractions"][1], 1.0)
        # Third prompt: Invalid prompt, gets a default score of 1.0.
        self.assertEqual(metrics["per_prompt_following_fractions"][2], 1.0)
        self.assertEqual(metrics["binary_prompt_satisfaction_rate"], 1.0)
        # Both identifiable prompts are exactly satisfied.
        self.assertEqual(metrics["binary_object_number_satisfaction_rate"], 1.0)
        self.assertEqual(metrics["binary_object_name_satisfaction_rate"], 1.0)
        # First prompt: exactly 2 objects (satisfied).
        self.assertEqual(metrics["per_prompt_binary_satisfaction"][0], 1)
        # Second prompt: exactly 2 apples (satisfied).
        self.assertEqual(metrics["per_prompt_binary_satisfaction"][1], 1)
        # Third prompt: invalid prompt (default to satisfied).
        self.assertEqual(metrics["per_prompt_binary_satisfaction"][2], 1)

        # Test case 4: Subset prompt.
        prompts = ["A scene with two apples and some other objects."] * 2
        scenes = torch.tensor(
            [
                # Scene has 3 apples (more than required).
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # apple
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # apple
                    [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # apple
                ],
                # Scene has 1 apple and 1 bowl (should have 2 apples).
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # apple
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # bowl
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # empty
                ],
            ]
        )

        metrics = compute_prompt_following_metrics(
            scene_vec_desc=scene_vec_desc, prompts=prompts, scenes=scenes
        )

        self.assertEqual(metrics["num_identifiable_prompts"], 2)
        self.assertEqual(metrics["identifiable_prompt_fraction"], 1.0)
        # First scene: 3/3 correct (all apples, more than required).
        # Second scene: 1/2 correct (1 apple out of 2 objects, missing 1 apple).
        # Average: (3/3 + 1/2) / 2 = (1.0 + 0.5) / 2 = 0.75.
        self.assertAlmostEqual(metrics["prompt_following_fraction"], 0.75, places=5)
        self.assertAlmostEqual(
            metrics["object_name_prompt_following_fraction"], 0.75, places=5
        )
        # First scene correct, 2nd half correct.
        self.assertAlmostEqual(
            metrics["per_prompt_following_fractions"][0], 1.0, places=5
        )
        self.assertAlmostEqual(
            metrics["per_prompt_following_fractions"][1], 0.5, places=5
        )
        # First scene satisfies the prompt (3 apples > 2 required).
        # Second scene doesn't satisfy the prompt (only 1 apple, not 2).
        # So 1/2 scenes satisfy the prompt.
        self.assertEqual(metrics["binary_prompt_satisfaction_rate"], 0.5)
        self.assertEqual(metrics["binary_object_name_satisfaction_rate"], 0.5)
        self.assertEqual(metrics["per_prompt_binary_satisfaction"][0], 1.0)
        self.assertEqual(metrics["per_prompt_binary_satisfaction"][1], 0)

    def test_compute_welded_object_pose_deviation_metric(self):
        # Define the SceneVecDescription with welded objects.
        scene_vec_desc = SceneVecDescription(
            drake_package_map=PackageMap(),
            static_directive=None,
            translation_vec_len=3,
            rotation_parametrization=RotationParametrization.AXIS_ANGLE,
            model_paths=[
                "tests/models/table.sdf",  # welded object
                "tests/models/box.sdf",  # non-welded object
                "tests/models/shelf.sdf",  # welded object
            ],
            model_path_vec_len=4,  # table, box, shelf, and [empty]
            welded_object_model_paths=[
                "tests/models/table.sdf",
                "tests/models/shelf.sdf",
            ],
        )

        # Dataset scenes with welded objects at specific poses.
        dataset_scenes = torch.tensor(
            [
                # Scene 1: table at (0,0,0), shelf at (1,1,0)
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # table
                    [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # box
                    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # shelf
                ],
                # Scene 2: table at (0,0,0), shelf at (1,1,0)
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # table
                    [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # box
                    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # shelf
                ],
            ]
        )

        # Synthesized scenes with welded objects at slightly different poses.
        synthesized_scenes = torch.tensor(
            [
                # Scene 1: table at (0.1,0,0), shelf at (1.1,1,0)
                [
                    [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # table
                    [0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # box
                    [1.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # shelf
                ],
                # Scene 2: table at (0.1,0,0), shelf at (1.1,1,0)
                [
                    [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # table
                    [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # box
                    [1.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # shelf
                ],
            ]
        )

        # Compute welded object pose deviation.
        pose_deviation = compute_welded_object_pose_deviation_metric(
            dataset_scenes=dataset_scenes,
            dataset_scene_vec_desc=scene_vec_desc,
            synthesized_scenes=synthesized_scenes,
            synthesized_scene_vec_desc=scene_vec_desc,
        )

        # Expected deviations:
        # Table: mean dataset pose (0,0,0) vs mean synth pose (0.1,0,0) -> L2 = 0.1
        # Shelf: mean dataset pose (1,1,0) vs mean synth pose (1.1,1,0) -> L2 = 0.1
        # Average deviation = (0.1 + 0.1) / 2 = 0.1
        self.assertAlmostEqual(pose_deviation, 0.1, places=6)

        # Test with no welded objects.
        scene_vec_desc_no_welded = SceneVecDescription(
            drake_package_map=PackageMap(),
            static_directive=None,
            translation_vec_len=3,
            rotation_parametrization=RotationParametrization.AXIS_ANGLE,
            model_paths=["tests/models/box.sdf"],
            model_path_vec_len=2,  # box and [empty]
            welded_object_model_paths=[],
        )

        pose_deviation = compute_welded_object_pose_deviation_metric(
            dataset_scenes=dataset_scenes,
            dataset_scene_vec_desc=scene_vec_desc_no_welded,
            synthesized_scenes=synthesized_scenes,
            synthesized_scene_vec_desc=scene_vec_desc_no_welded,
        )

        # Should return 0.0 when there are no welded objects.
        self.assertEqual(pose_deviation, 0.0)

        # Test with welded objects in dataset but not synthesized.
        synthesized_scenes_no_welded = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # empty
            ]
        )

        pose_deviation = compute_welded_object_pose_deviation_metric(
            dataset_scenes=dataset_scenes,
            dataset_scene_vec_desc=scene_vec_desc,
            synthesized_scenes=synthesized_scenes_no_welded,
            synthesized_scene_vec_desc=scene_vec_desc,
        )

        # Should return infinity when there are welded objects in dataset but not
        # synthesized.
        self.assertEqual(pose_deviation, torch.inf)

        # Test with welded objects in synthesized but not dataset.
        dataset_scenes_no_welded = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # empty
            ]
        )

        pose_deviation = compute_welded_object_pose_deviation_metric(
            dataset_scenes=dataset_scenes_no_welded,
            dataset_scene_vec_desc=scene_vec_desc,
            synthesized_scenes=synthesized_scenes,
            synthesized_scene_vec_desc=scene_vec_desc,
        )

        # Should return infinity when there are welded objects in synthesized but not
        # dataset.
        self.assertEqual(pose_deviation, torch.inf)


if __name__ == "__main__":
    unittest.main()
