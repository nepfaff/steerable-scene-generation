import json
import os
import subprocess
import unittest

from typing import List

from tests.integration.common import (
    extract_output_dir_from_stdout,
    find_first_ckpt_file,
)


class TestMainScriptWithConfigs(unittest.TestCase):
    def setUp(self):
        # Load the mock dataset metadata.
        mock_dataset_path = "tests/datasets/mock_dataset"
        metadata_path = os.path.join(mock_dataset_path, "metadata.json")
        with open(metadata_path, "r") as metadata_file:
            metadata = json.load(metadata_file)

        # Define default arguments for the script.
        self.default_args = [
            "python",
            "main.py",
            "+name=test",
            "wandb.mode=offline",
            "experiment.training.batch_size=1",
            "experiment.validation.batch_size=1",
            "experiment.test.batch_size=1",
            "experiment.training.max_epochs=-1",
            "experiment.training.max_steps=1",
            "experiment.validation.val_every_n_step=1",
            "experiment.training.precision=32",
            "experiment.validation.precision=32",
            "experiment.test.precision=32",
            f"dataset.processed_scene_data_path={mock_dataset_path}",
            f"dataset.max_num_objects_per_scene={metadata['max_num_objects_per_scene']}",
            f"dataset.translation_vec_len={metadata['translation_vec_len']}",
            f"dataset.rotation_parametrization={metadata['rotation_parametrization']}",
            f"dataset.model_path_vec_len={metadata['model_path_vec_len']}",
            "dataset.drake_package_maps=[]",
            "dataset.static_directive=null",
            "dataset.val_ratio=0.1",
            "dataset.test_ratio=0.1",
            "algorithm=scene_diffuser_flux_transformer",
            "algorithm.trainer=ddpm",
            "algorithm.validation.num_samples_to_render=1",
            "algorithm.validation.num_samples_to_visualize=0",
            "algorithm.validation.num_directives_to_generate=0",
            "algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=1",
            "algorithm.noise_schedule.num_train_timesteps=2",
            "algorithm.noise_schedule.ddim.num_inference_timesteps=2",
        ]
        self.default_args_with_training = [
            *self.default_args,
            "experiment.tasks=[training]",
        ]
        self.default_args_with_training_test_and_predict = [
            *self.default_args,
            "experiment.tasks=[training, test, predict]",
        ]
        self.disable_sampling_args = [
            "algorithm.test.num_samples_to_render=0",
            "algorithm.test.num_samples_to_render_as_label=0",
            "algorithm.test.num_samples_to_visualize=0",
            "algorithm.test.num_directives_to_generate=0",
            "algorithm.test.num_samples_to_save_as_pickle=0",
        ]
        self.test_sampling_args = [
            "experiment.test.batch_size=2",  # Different from sample number
            "algorithm.test.num_samples_to_render=1",
            "algorithm.test.num_samples_to_render_as_label=1",
            "algorithm.test.num_samples_to_visualize=1",
            "algorithm.test.num_directives_to_generate=1",
            "algorithm.test.num_samples_to_save_as_pickle=1",
        ]
        self.classifier_free_guidance_enable_args = [
            "algorithm.classifier_free_guidance.use=True",
        ]
        self.classifier_free_guidance_disable_args = [
            "algorithm.classifier_free_guidance.use=False",
        ]
        self.small_model_config = [
            "algorithm=scene_diffuser_flux_transformer",
            "algorithm.model.hidden_dim=16",
            "algorithm.model.mlp_ratio=1",
            "algorithm.model.num_single_layers=1",
            "algorithm.model.num_double_layers=0",
            "algorithm.model.num_heads=2",
            "algorithm.model.head_dim=8",
            "algorithm.classifier_free_guidance.txt_encoder=bert",
            "algorithm.classifier_free_guidance.txt_encoder_size=tiny",
            "algorithm.classifier_free_guidance.txt_encoder_coarse=null",
        ]

    def run_command(self, command: List[str]) -> subprocess.CompletedProcess:
        # Run the command as a subprocess.
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        return result

    def get_ckpt_path_from_stdout(self, stdout: str) -> str:
        output_dir = extract_output_dir_from_stdout(stdout)
        self.assertIsNotNone(output_dir)

        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        self.assertTrue(os.path.exists(checkpoint_dir))

        ckpt_path = find_first_ckpt_file(checkpoint_dir)
        self.assertIsNotNone(ckpt_path)

        return ckpt_path

    def test_main_script_with_default_config(self):
        # Define the command to run.
        command = self.default_args + self.disable_sampling_args

        result = self.run_command(command)

        # Assert that the script executed without errors.
        self.assertEqual(
            result.returncode, 0, f"Script failed with stderr: {result.stderr}"
        )

    def test_scene_diffuser_flux_transformer(self):
        # Use DDPM as already have an explicit FM test with this model.
        command = [
            *self.default_args_with_training_test_and_predict,
            *self.disable_sampling_args,
            "algorithm=scene_diffuser_flux_transformer",
            "algorithm.trainer=ddpm",
        ]

        for classifier_free_guidance_args in [
            self.classifier_free_guidance_enable_args,
            self.classifier_free_guidance_disable_args,
        ]:
            command_with_classifier_free_guidance = [
                *command,
                *classifier_free_guidance_args,
            ]
            result = self.run_command(command_with_classifier_free_guidance)
            self.assertEqual(
                result.returncode, 0, f"Script failed with stderr: {result.stderr}"
            )

    def test_scene_diffuser_mixed_flux_transformer(self):
        command = [
            *self.default_args_with_training_test_and_predict,
            *self.disable_sampling_args,
            "algorithm=scene_diffuser_mixed_flux_transformer",
            "algorithm.trainer=mixed",
        ]

        for classifier_free_guidance_args in [
            self.classifier_free_guidance_enable_args,
            self.classifier_free_guidance_disable_args,
        ]:
            command_with_classifier_free_guidance = [
                *command,
                *classifier_free_guidance_args,
            ]
            result = self.run_command(command_with_classifier_free_guidance)
            self.assertEqual(
                result.returncode, 0, f"Script failed with stderr: {result.stderr}"
            )

    def test_scene_diffuser_diffuscene(self):
        command = [
            *self.default_args_with_training_test_and_predict,
            *self.disable_sampling_args,
            "algorithm=scene_diffuser_diffuscene",
        ]

        for classifier_free_guidance_args in [
            self.classifier_free_guidance_enable_args,
            self.classifier_free_guidance_disable_args,
        ]:
            command_with_classifier_free_guidance = [
                *command,
                *classifier_free_guidance_args,
            ]
            result = self.run_command(command_with_classifier_free_guidance)
            self.assertEqual(
                result.returncode, 0, f"Script failed with stderr: {result.stderr}"
            )

    def test_scene_diffuser_mixed_diffuscene(self):
        command = [
            *self.default_args_with_training_test_and_predict,
            *self.disable_sampling_args,
            "algorithm=scene_diffuser_mixed_diffuscene",
            "algorithm.trainer=mixed",
        ]

        for classifier_free_guidance_args in [
            self.classifier_free_guidance_enable_args,
            self.classifier_free_guidance_disable_args,
        ]:
            command_with_classifier_free_guidance = [
                *command,
                *classifier_free_guidance_args,
            ]
            result = self.run_command(command_with_classifier_free_guidance)
            self.assertEqual(
                result.returncode, 0, f"Script failed with stderr: {result.stderr}"
            )

    def test_scene_diffuser_midiffusion(self):
        command = [
            *self.default_args_with_training_test_and_predict,
            *self.disable_sampling_args,
            "algorithm=scene_diffuser_midiffusion",
        ]

        for classifier_free_guidance_args in [
            self.classifier_free_guidance_enable_args,
            self.classifier_free_guidance_disable_args,
        ]:
            command_with_classifier_free_guidance = [
                *command,
                *classifier_free_guidance_args,
            ]
            result = self.run_command(command_with_classifier_free_guidance)
            self.assertEqual(
                result.returncode, 0, f"Script failed with stderr: {result.stderr}"
            )

    def test_scene_diffuser_mixed_midiffusion(self):
        command = [
            *self.default_args_with_training_test_and_predict,
            *self.disable_sampling_args,
            "algorithm=scene_diffuser_mixed_midiffusion",
            "algorithm.trainer=mixed",
        ]

        for classifier_free_guidance_args in [
            self.classifier_free_guidance_enable_args,
            self.classifier_free_guidance_disable_args,
        ]:
            command_with_classifier_free_guidance = [
                *command,
                *classifier_free_guidance_args,
            ]
            result = self.run_command(command_with_classifier_free_guidance)
            self.assertEqual(
                result.returncode, 0, f"Script failed with stderr: {result.stderr}"
            )

    def test_rl_ppo(self):
        # Use a tiny model to make sampling fast.
        command = [
            *self.default_args_with_training_test_and_predict,
            *self.disable_sampling_args,
            *self.small_model_config,
            "algorithm.trainer=rl_ppo",
            "algorithm.ddpo.use_prompt_following_reward=True",
            "algorithm.ddpo.last_n_timesteps_only=2",
            "algorithm.ddpo.batch_size=2",
            "algorithm.noise_schedule.num_train_timesteps=2",
            "algorithm.noise_schedule.ddim.num_inference_timesteps=2",
        ]

        for classifier_free_guidance_args in [
            self.classifier_free_guidance_enable_args,
            self.classifier_free_guidance_disable_args,
        ]:
            command_with_classifier_free_guidance = [
                *command,
                *classifier_free_guidance_args,
            ]
            result = self.run_command(command_with_classifier_free_guidance)
            self.assertEqual(
                result.returncode, 0, f"Script failed with stderr: {result.stderr}"
            )

    def test_rl_score(self):
        # Use a tiny model to make sampling fast.
        command = [
            *self.default_args_with_training_test_and_predict,
            *self.disable_sampling_args,
            *self.small_model_config,
            "algorithm.trainer=rl_score",
            "algorithm.ddpo.use_non_penetration_reward=True",
            "algorithm.ddpo.last_n_timesteps_only=2",
            "algorithm.ddpo.batch_size=2",
            "algorithm.ddpo.ppo.num_epochs=1",
            "algorithm.noise_schedule.num_train_timesteps=2",
            "algorithm.noise_schedule.ddim.num_inference_timesteps=2",
        ]

        for classifier_free_guidance_args in [
            self.classifier_free_guidance_enable_args,
            self.classifier_free_guidance_disable_args,
        ]:
            command_with_classifier_free_guidance = [
                *command,
                *classifier_free_guidance_args,
            ]
            result = self.run_command(command_with_classifier_free_guidance)
            self.assertEqual(
                result.returncode, 0, f"Script failed with stderr: {result.stderr}"
            )

    def test_rl_score_with_regularization(self):
        # Use a tiny model to make sampling fast.
        command = [
            *self.default_args_with_training_test_and_predict,
            *self.disable_sampling_args,
            *self.small_model_config,
            "algorithm.trainer=rl_score",
            "algorithm.ddpo.use_object_number_reward=True",
            "algorithm.ddpo.last_n_timesteps_only=2",
            "algorithm.ddpo.batch_size=2",
            "algorithm.ddpo.ppo.num_epochs=1",
            "algorithm.noise_schedule.num_train_timesteps=2",
            "algorithm.noise_schedule.ddim.num_inference_timesteps=2",
            "algorithm.ddpo.ddpm_reg_weight=1.0",
        ]

        for classifier_free_guidance_args in [
            self.classifier_free_guidance_enable_args,
            self.classifier_free_guidance_disable_args,
        ]:
            command_with_classifier_free_guidance = [
                *command,
                *classifier_free_guidance_args,
            ]
            result = self.run_command(command_with_classifier_free_guidance)
            self.assertEqual(
                result.returncode, 0, f"Script failed with stderr: {result.stderr}"
            )

    def test_sampling(self):
        # Use a tiny model to make sampling fast.
        command = [
            *self.default_args_with_training_test_and_predict,
            *self.test_sampling_args,
            *self.small_model_config,
        ]

        for classifier_free_guidance_args in [
            self.classifier_free_guidance_enable_args,
            self.classifier_free_guidance_disable_args,
        ]:
            command_with_classifier_free_guidance = [
                *command,
                *classifier_free_guidance_args,
            ]
            result = self.run_command(command_with_classifier_free_guidance)
            self.assertEqual(
                result.returncode, 0, f"Script failed with stderr: {result.stderr}"
            )
            self.assertIn("Test metric", result.stdout)

    def test_sampling_with_ddim(self):
        # Use a tiny model to make sampling fast.
        command = [
            *self.default_args_with_training_test_and_predict,
            *self.test_sampling_args,
            *self.small_model_config,
            "algorithm.noise_schedule.scheduler=ddim",
        ]

        for classifier_free_guidance_args in [
            self.classifier_free_guidance_enable_args,
            self.classifier_free_guidance_disable_args,
        ]:
            command_with_classifier_free_guidance = [
                *command,
                *classifier_free_guidance_args,
            ]
            result = self.run_command(command_with_classifier_free_guidance)
            self.assertEqual(
                result.returncode, 0, f"Script failed with stderr: {result.stderr}"
            )
            self.assertIn("Test metric", result.stdout)

    def test_continuous_only(self):
        # Use a tiny model to make sampling fast.
        command = [
            *self.default_args_with_training_test_and_predict,
            *self.test_sampling_args,
            *self.small_model_config,
            "algorithm.continuous_discrete_only.continuous_only=True",
        ]

        for classifier_free_guidance_args in [
            self.classifier_free_guidance_enable_args,
            self.classifier_free_guidance_disable_args,
        ]:
            command_with_classifier_free_guidance = [
                *command,
                *classifier_free_guidance_args,
            ]
            result = self.run_command(command_with_classifier_free_guidance)
            self.assertEqual(
                result.returncode, 0, f"Script failed with stderr: {result.stderr}"
            )
            self.assertIn("Test metric", result.stdout)

    def test_discrete_only(self):
        # Use a tiny model to make sampling fast.
        command = [
            *self.default_args_with_training_test_and_predict,
            *self.test_sampling_args,
            *self.small_model_config,
            "algorithm.continuous_discrete_only.discrete_only=True",
        ]

        for classifier_free_guidance_args in [
            self.classifier_free_guidance_enable_args,
            self.classifier_free_guidance_disable_args,
        ]:
            command_with_classifier_free_guidance = [
                *command,
                *classifier_free_guidance_args,
            ]
            result = self.run_command(command_with_classifier_free_guidance)
            self.assertEqual(
                result.returncode, 0, f"Script failed with stderr: {result.stderr}"
            )
            self.assertIn("Test metric", result.stdout)

    def test_post_processing(self):
        # Use a tiny model to make sampling fast.
        command = [
            *self.default_args_with_training_test_and_predict,
            *self.test_sampling_args,
            *self.classifier_free_guidance_disable_args,
            *self.small_model_config,
            "algorithm.postprocessing.apply_non_penetration_projection=True",
            "algorithm.postprocessing.apply_forward_simulation=True",
            "algorithm.postprocessing.non_penetration_projection.iteration_limit=10",
            "algorithm.postprocessing.forward_simulation.simulation_time_s=0.01",
            "algorithm.postprocessing.forward_simulation.timeout_s=1.0",
        ]

        result = self.run_command(command)
        self.assertEqual(
            result.returncode, 0, f"Script failed with stderr: {result.stderr}"
        )

    def test_checkpoint_saving_and_loading(self):
        # Use a tiny model for speed.
        command = [
            *self.default_args_with_training,
            *self.disable_sampling_args,
            *self.small_model_config,
            "experiment.training.checkpointing.every_n_train_steps=1",
            "experiment.training.checkpointing.every_n_epochs=null",
            "experiment.training.max_steps=2",  # Required to save checkpoint
        ]

        for classifier_free_guidance_args in [
            self.classifier_free_guidance_enable_args,
            self.classifier_free_guidance_disable_args,
        ]:
            # Save checkpoint.
            command_with_classifier_free_guidance = [
                *command,
                *classifier_free_guidance_args,
            ]
            result = self.run_command(command_with_classifier_free_guidance)
            self.assertEqual(
                result.returncode, 0, f"Script failed with stderr: {result.stderr}"
            )
            self.assertIn("Loaded normalizer state from the dataset.", result.stdout)

            # Extract the checkpoint path from the stdout.
            ckpt_path = self.get_ckpt_path_from_stdout(result.stdout)

            # Test load checkpoint.
            command_with_classifier_free_guidance = [
                *command,
                *classifier_free_guidance_args,
                f"load='{ckpt_path}'",
            ]
            result = self.run_command(command_with_classifier_free_guidance)
            self.assertEqual(
                result.returncode, 0, f"Script failed with stderr: {result.stderr}"
            )
            self.assertIn("Will load checkpoint", result.stdout)
            self.assertIn("Loaded normalizer state from the checkpoint.", result.stdout)

    def test_main_script_with_invalid_algo(self):
        # Test with an invalid algo.
        command = [*self.default_args, "algorithm=non_existent_algorithm"]

        result = self.run_command(command)

        # Assert that the script fails as expected.
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "Could not find 'algorithm/non_existent_algorithm'",
            result.stderr,
            f"Unexpected stderr: {result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
