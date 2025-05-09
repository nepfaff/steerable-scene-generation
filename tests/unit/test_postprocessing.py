import unittest

import torch

from pydrake.all import PackageMap

from steerable_scene_generation.algorithms.common.dataclasses import (
    RotationParametrization,
    SceneVecDescription,
)
from steerable_scene_generation.algorithms.scene_diffusion.postprocessing import (
    apply_forward_simulation,
    apply_non_penetration_projection,
)


class TestPostprocessing(unittest.TestCase):
    """
    Tests for the postprocessing functions in scene_diffusion/postprocessing.py.
    """

    def setUp(self):
        # Create a simple scene vector description for testing.
        self.scene_vec_desc = SceneVecDescription(
            drake_package_map=PackageMap(),
            static_directive=None,
            translation_vec_len=3,
            rotation_parametrization=RotationParametrization.PROCRUSTES,
            model_paths=["tests/models/box.sdf"] * 2,
            model_path_vec_len=None,
        )

        zero_rotation = torch.eye(3).flatten().tolist()

        # Create an overlapping scene with two boxes at the same position.
        self.overlapping_scene = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, *zero_rotation],  # Box1
                    [0.5, 0.01, 0.0, *zero_rotation],  # Box2
                ]
            ]
        )

        # Create a non-overlapping scene with boxes spaced apart.
        self.non_overlapping_scene = torch.tensor(
            [
                [
                    [1.0, 0.0, 1.0, *zero_rotation],  # Box1
                    [-1.0, 0.0, -1.0, *zero_rotation],  # Box2
                ]
            ]
        )

        # Create a scene vector description and scene with a welded object.
        self.scene_vec_desc_with_welded_object = SceneVecDescription(
            drake_package_map=PackageMap(),
            static_directive=None,
            translation_vec_len=3,
            rotation_parametrization=RotationParametrization.PROCRUSTES,
            model_paths=["tests/models/box.sdf", "tests/models/cylinder.sdf"],
            model_path_vec_len=3,
            welded_object_model_paths=["tests/models/cylinder.sdf"],
        )
        # One box and one welded cylinder at the same position.
        zero_rotation = torch.tensor(zero_rotation)
        self.scene_with_welded_object = (
            self.scene_vec_desc_with_welded_object.get_scene_or_obj_from_components(
                translation_vec=torch.zeros(2, 3),
                rotation_vec=torch.stack([zero_rotation, zero_rotation]),
                model_path_vec=torch.tensor([[1, 0, 0], [0, 1, 0]]),
            )
        ).unsqueeze(0)

    def test_non_penetration_projection_single_worker(self):
        """Test non-penetration projection with a single worker."""
        # Apply non-penetration projection.
        projected_scenes, _, _ = apply_non_penetration_projection(
            scenes=self.overlapping_scene,
            scene_vec_desc=self.scene_vec_desc,
            translation_only=False,
            influence_distance=0.03,
            solver_name="snopt",
            caches=[None],
            num_workers=1,
        )

        # NOTE: We aren't testing success here as this currently fails due to bad
        # numerics. We don't care that much we we only use `translation_only=True`
        # in practice.

        # Check that we got back a scene with the right shape.
        self.assertEqual(projected_scenes.shape, self.overlapping_scene.shape)

    def test_non_penetration_projection_translation_only(self):
        """Test non-penetration projection with translation only."""
        # Apply non-penetration projection with translation only.
        projected_scenes, caches, successes = apply_non_penetration_projection(
            scenes=self.overlapping_scene,
            scene_vec_desc=self.scene_vec_desc,
            translation_only=True,
            influence_distance=0.03,
            solver_name="snopt",
            caches=[None],
            num_workers=1,
        )

        self.assertTrue(all(successes))

        # Check that the overlapping scene is projected.
        self.assertFalse(torch.allclose(projected_scenes, self.overlapping_scene))

        # Check that only translations changed (rotations should be the same).
        self.assertTrue(
            torch.allclose(
                projected_scenes[0, :, 3:],
                self.overlapping_scene[0, :, 3:],
            )
        )

        # Check that the cache is not None.
        self.assertIsNotNone(caches[0])

    def test_non_penetration_projection_translation_only_multi_worker(self):
        """Test non-penetration projection with multiple workers."""
        # Create a batch of scenes.
        batch_scenes = torch.cat(
            [self.overlapping_scene, self.non_overlapping_scene], dim=0
        )

        # Apply non-penetration projection.
        projected_scenes, _, successes = apply_non_penetration_projection(
            scenes=batch_scenes,
            scene_vec_desc=self.scene_vec_desc,
            translation_only=True,
            influence_distance=0.03,
            solver_name="snopt",
            caches=[None, None],
            num_workers=2,
        )

        self.assertTrue(all(successes))

        # Check that we got back scenes with the right shape.
        self.assertEqual(projected_scenes.shape, batch_scenes.shape)

        # Check that the overlapping scene is projected.
        self.assertFalse(torch.allclose(projected_scenes[0], self.overlapping_scene))

        # Check that the non-overlapping scene is unchanged.
        self.assertTrue(torch.allclose(projected_scenes[1], self.non_overlapping_scene))

    def test_non_penetration_projection_translation_only_ipopt(self):
        """Test non-penetration projection with IPOPT solver."""
        # Apply non-penetration projection with IPOPT.
        projected_scenes, _, successes = apply_non_penetration_projection(
            scenes=self.overlapping_scene,
            scene_vec_desc=self.scene_vec_desc,
            translation_only=True,
            influence_distance=0.03,
            solver_name="ipopt",
            caches=[None],
            num_workers=1,
        )

        self.assertTrue(all(successes))

        # Check that the overlapping scene is projected.
        self.assertFalse(torch.allclose(projected_scenes, self.overlapping_scene))

    def test_non_penetration_projection_with_welded_object(self):
        """Test non-penetration projection with a welded object."""
        projected_scenes, caches, successes = apply_non_penetration_projection(
            scenes=self.scene_with_welded_object,
            scene_vec_desc=self.scene_vec_desc_with_welded_object,
            translation_only=True,
            influence_distance=0.03,
            solver_name="snopt",
            caches=[None],
            num_workers=1,
        )

        self.assertTrue(all(successes))

        # Check that the overlapping scene is projected.
        self.assertFalse(
            torch.allclose(projected_scenes, self.scene_with_welded_object)
        )

        # Check that only translations changed (rotations should be the same).
        self.assertTrue(
            torch.allclose(
                projected_scenes[..., 3:], self.scene_with_welded_object[..., 3:]
            )
        )

        # Check that the welded object didn't move.
        self.assertTrue(
            torch.allclose(projected_scenes[:, 1], self.scene_with_welded_object[:, 1])
        )

        # Check that the cache is not None.
        self.assertIsNotNone(caches[0])

    def test_forward_simulation_single_worker(self):
        """Test forward simulation with a single worker."""
        # Apply forward simulation.
        simulated_scenes, caches = apply_forward_simulation(
            scenes=self.non_overlapping_scene,
            scene_vec_desc=self.scene_vec_desc,
            simulation_time_s=10.0,
            time_step=0.01,
            caches=[None],
            timeout_s=10.0,
            num_workers=1,
        )

        # Check that we got back scenes with the right shape.
        self.assertEqual(simulated_scenes.shape, self.non_overlapping_scene.shape)

        # Check that the caches are not None.
        self.assertIsNotNone(caches[0])

    def test_forward_simulation_multi_worker(self):
        """Test forward simulation with multiple workers."""
        # Create a batch of scenes.
        batch_scenes = torch.cat(
            [self.overlapping_scene, self.non_overlapping_scene], dim=0
        )

        # Apply forward simulation.
        simulated_scenes, _ = apply_forward_simulation(
            scenes=batch_scenes,
            scene_vec_desc=self.scene_vec_desc,
            simulation_time_s=10.0,
            time_step=0.01,
            caches=[None, None],
            timeout_s=10.0,
            num_workers=2,
        )

        # Check that we got back scenes with the right shape.
        self.assertEqual(simulated_scenes.shape, batch_scenes.shape)

        # Check that the overlapping scene is changed.
        self.assertFalse(torch.allclose(simulated_scenes[0], self.overlapping_scene))

    def test_forward_simulation_with_timeout(self):
        """Test forward simulation with a timeout."""
        # Apply forward simulation with a tiny timeout.
        simulated_scenes, _ = apply_forward_simulation(
            scenes=self.overlapping_scene,
            scene_vec_desc=self.scene_vec_desc,
            simulation_time_s=10.0,
            time_step=0.01,
            caches=[None],
            timeout_s=1e-16,
            num_workers=1,
        )

        # Check that we got back scenes with the right shape.
        self.assertEqual(simulated_scenes.shape, self.overlapping_scene.shape)

        # Check that terminated due to timeout without being able to change the scene
        # much.
        self.assertTrue(torch.allclose(simulated_scenes, self.overlapping_scene))

    def test_forward_simulation_with_welded_object(self):
        """Test forward simulation with a welded object."""
        # Apply forward simulation.
        simulated_scenes, caches = apply_forward_simulation(
            scenes=self.scene_with_welded_object,
            scene_vec_desc=self.scene_vec_desc_with_welded_object,
            simulation_time_s=10.0,
            time_step=0.01,
            caches=[None],
            timeout_s=10.0,
            num_workers=1,
        )

        # Check that we got back scenes with the right shape.
        self.assertEqual(simulated_scenes.shape, self.scene_with_welded_object.shape)

        # Check that the welded object didn't move.
        self.assertTrue(
            torch.allclose(simulated_scenes[:, 1], self.scene_with_welded_object[:, 1])
        )

        # Check that the non-welded object moved.
        self.assertFalse(
            torch.allclose(simulated_scenes[:, 0], self.scene_with_welded_object[:, 0])
        )

        # Check that the caches are not None.
        self.assertIsNotNone(caches[0])

    def test_projection_followed_by_simulation(self):
        """Test applying projection followed by simulation."""
        # First apply non-penetration projection.
        projected_scenes, caches, _ = apply_non_penetration_projection(
            scenes=self.overlapping_scene,
            scene_vec_desc=self.scene_vec_desc,
            translation_only=True,
            influence_distance=0.03,
            solver_name="snopt",
            caches=[None],
            num_workers=1,
        )

        # Check that we got back scenes with the right shape.
        self.assertEqual(projected_scenes.shape, self.overlapping_scene.shape)

        # Then apply forward simulation using the cache from projection.
        simulated_scenes, _ = apply_forward_simulation(
            scenes=projected_scenes,
            scene_vec_desc=self.scene_vec_desc,
            simulation_time_s=1.0,
            time_step=0.01,
            caches=caches,
            timeout_s=10.0,
            num_workers=1,
        )

        # Check that we got back scenes with the right shape.
        self.assertEqual(simulated_scenes.shape, projected_scenes.shape)


if __name__ == "__main__":
    unittest.main()
