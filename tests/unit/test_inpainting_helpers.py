import unittest

import torch

from pydrake.all import PackageMap

from steerable_scene_generation.algorithms.common.dataclasses import (
    RotationParametrization,
    SceneVecDescription,
)
from steerable_scene_generation.algorithms.scene_diffusion.inpainting_helpers import (
    generate_empty_object_inpainting_masks,
    generate_non_penetration_inpainting_masks,
    generate_physical_feasibility_inpainting_masks,
    generate_static_equilibrium_inpainting_masks_with_heuristic,
)


class TestInpaintingHelpers(unittest.TestCase):
    def test_generate_empty_object_inpainting_masks(self):
        scene_vec_desc = SceneVecDescription(
            drake_package_map=PackageMap(),
            static_directive=None,
            translation_vec_len=3,
            rotation_parametrization=RotationParametrization.AXIS_ANGLE,
            model_paths=["tests/models/box.sdf", "tests/models/sphere.sdf"],
            model_path_vec_len=3,  # Box, sphere, and [empty]
        )

        # Create a batch of scenes with some empty objects.
        scenes = torch.tensor(
            [
                # Scene 1: One box, one empty, one sphere.
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # Box
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Empty
                    [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # Sphere
                ],
                # Scene 2: Two empty objects, one box.
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Empty
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Empty
                    [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # Box
                ],
            ]
        )

        # Generate masks
        masks, empty_object_numbers = generate_empty_object_inpainting_masks(
            scenes=scenes, scene_vec_desc=scene_vec_desc
        )

        # Check mask shapes.
        self.assertEqual(masks.shape, scenes.shape)
        self.assertEqual(len(empty_object_numbers), 2)

        # Check mask values for scene 1.
        self.assertFalse(masks[0, 0].any())  # Box should not be masked
        self.assertTrue(masks[0, 1].all())  # Empty object should be masked
        self.assertFalse(masks[0, 2].any())  # Sphere should not be masked
        self.assertEqual(empty_object_numbers[0], 1)  # One empty object

        # Check mask values for scene 2.
        self.assertTrue(masks[1, 0].all())  # Empty object should be masked
        self.assertTrue(masks[1, 1].all())  # Empty object should be masked
        self.assertFalse(masks[1, 2].any())  # Box should not be masked
        self.assertEqual(empty_object_numbers[1], 2)  # Two empty objects

    def test_generate_non_penetration_inpainting_masks(self):
        scene_vec_desc = SceneVecDescription(
            drake_package_map=PackageMap(),
            static_directive=None,
            translation_vec_len=3,
            rotation_parametrization=RotationParametrization.AXIS_ANGLE,
            model_paths=["tests/models/box.sdf"] * 2,
            model_path_vec_len=None,
        )

        # Create a batch of scenes with and without penetration.
        scenes = torch.tensor(
            [
                # Scene 1: No penetration (boxes are >1.0 unit apart).
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Box1
                    [1.001, 0.0, 0.0, 0.0, 0.0, 0.0],  # Box2
                ],
                # Scene 2: With penetration (boxes are 0.5 unit apart, which causes
                # penetration).
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Box1
                    [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],  # Box2
                ],
            ]
        )

        # Generate masks.
        masks, penetration_distances = generate_non_penetration_inpainting_masks(
            scenes=scenes, scene_vec_desc=scene_vec_desc, threshold=0.0
        )

        # Check mask shapes.
        self.assertEqual(masks.shape, scenes.shape)
        self.assertEqual(len(penetration_distances), 2)

        # Check mask values for scene 1 (no penetration).
        self.assertFalse(masks[0, 0].any())  # Box1 should not be masked
        self.assertFalse(masks[0, 1].any())  # Box2 should not be masked
        self.assertEqual(penetration_distances[0], 0.0)  # No penetration

        # Check mask values for scene 2 (with penetration).
        self.assertTrue(masks[1, 0].any())  # Box1 should be masked
        self.assertTrue(masks[1, 1].any())  # Box2 should be masked
        self.assertGreater(
            penetration_distances[1], 0.0
        )  # Positive penetration distance

        # Test with a different threshold.
        masks, penetration_distances = generate_non_penetration_inpainting_masks(
            scenes=scenes, scene_vec_desc=scene_vec_desc, threshold=0.1
        )

        # With threshold 0.1, both scenes should have masks.
        self.assertTrue(masks.all())

    def test_generate_static_equilibrium_inpainting_masks_with_heuristic(self):
        scene_vec_desc = SceneVecDescription(
            drake_package_map=PackageMap(),
            static_directive=None,
            translation_vec_len=3,
            rotation_parametrization=RotationParametrization.AXIS_ANGLE,
            model_paths=["tests/models/box.sdf"] * 3,
            model_path_vec_len=None,
        )

        # Create a batch of scenes with different equilibrium states.
        scenes = torch.tensor(
            [
                # Scene 1: Objects in contact (stable).
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Box1
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Box2
                    [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Box3
                ],
                # Scene 2: One floating object (unstable).
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Box1
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Box2
                    [3.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Box3 (floating)
                ],
            ]
        )

        # Generate masks with a small distance threshold.
        (
            masks,
            non_static_counts,
        ) = generate_static_equilibrium_inpainting_masks_with_heuristic(
            scenes=scenes, scene_vec_desc=scene_vec_desc, distance_threshold=0.1
        )

        # Check mask shapes.
        self.assertEqual(masks.shape, scenes.shape)
        self.assertEqual(len(non_static_counts), 2)

        # For scene 1, no objects should be masked as they're all in contact.
        self.assertEqual(non_static_counts[0], 0)

        # For scene 2, Box3 should be masked as it's floating.
        self.assertGreater(non_static_counts[1], 0)
        self.assertTrue(masks[1, 2].any())  # Box3 should be masked

    def test_generate_physical_feasibility_inpainting_masks(self):
        scene_vec_desc = SceneVecDescription(
            drake_package_map=PackageMap(),
            static_directive=None,
            translation_vec_len=3,
            rotation_parametrization=RotationParametrization.AXIS_ANGLE,
            model_paths=["tests/models/box.sdf", "tests/models/sphere.sdf"],
            welded_object_model_paths=["tests/models/sphere.sdf"],
            model_path_vec_len=3,
        )

        # Create a batch of scenes with different physical feasibility issues.
        scenes = torch.tensor(
            [
                # Scene 1: Physically feasible (no penetration, stably resting on
                # static sphere).
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # Sphere1
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # Box1
                    [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # Box2
                ],
                # Scene 2: Penetration issue.
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # Box1 (penetrating)
                    [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # Box2 (penetrating)
                    [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # Box3
                ],
                # Scene 3: Static equilibrium issue.
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # Sphere1
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # Box1
                    [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # Box2 (floating)
                ],
            ]
        )

        def check_results(masks, penetration_distances, non_static_counts):
            # Check mask shapes.
            self.assertEqual(masks.shape, scenes.shape)
            self.assertEqual(len(penetration_distances), 3)
            self.assertEqual(len(non_static_counts), 3)

            # Scene 1: Should be physically feasible.
            self.assertEqual(penetration_distances[0], 0.0)
            self.assertEqual(non_static_counts[0], 0)

            # Scene 2: Should have penetration issues.
            self.assertGreater(penetration_distances[1], 0.0)
            self.assertTrue(
                masks[1, 0].all() and masks[1, 1].all()
            )  # Box1 and Box2 should be masked

            # Scene 3: Should have static equilibrium issues.
            self.assertEqual(penetration_distances[2], 0.0)
            self.assertGreater(non_static_counts[2], 0)
            self.assertTrue(masks[2, 2].all())  # Box3 should be masked

        # Generate masks using heuristic method.
        (
            masks,
            penetration_distances,
            non_static_counts,
        ) = generate_physical_feasibility_inpainting_masks(
            scenes=scenes,
            scene_vec_desc=scene_vec_desc,
            non_penetration_threshold=-1e-6,
            use_sim=False,
            static_equilibrium_distance_threshold=0.1,
        )
        check_results(masks, penetration_distances, non_static_counts)

        # Test with simulation-based method.
        (
            masks,
            penetration_distances,
            non_static_counts,
        ) = generate_physical_feasibility_inpainting_masks(
            scenes=scenes,
            scene_vec_desc=scene_vec_desc,
            non_penetration_threshold=-1e-6,
            use_sim=True,
            sim_duration=0.1,
            sim_time_step=0.01,
            sim_translation_threshold=0.01,
            sim_rotation_threshold=0.01,
        )
        check_results(masks, penetration_distances, non_static_counts)


if __name__ == "__main__":
    unittest.main()
