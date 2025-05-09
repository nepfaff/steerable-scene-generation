import json
import os
import subprocess
import unittest

from typing import List

from tests.integration.common import (
    extract_output_dir_from_stdout,
    find_first_ckpt_file,
)


class TestEvaluateCheckpointsScript(unittest.TestCase):
    def setUp(self):
        # Load the mock dataset metadata.
        mock_dataset_path = "tests/datasets/mock_dataset"
        metadata_path = os.path.join(mock_dataset_path, "metadata.json")
        with open(metadata_path, "r") as metadata_file:
            metadata = json.load(metadata_file)

        # Define arguments for speeding up the test.
        self.fast_args = [
            "wandb.mode=offline",
            "experiment.tasks=[training]",
            # Use mock dataset.
            f"dataset.processed_scene_data_path={mock_dataset_path}",
            f"dataset.max_num_objects_per_scene={metadata['max_num_objects_per_scene']}",
            f"dataset.translation_vec_len={metadata['translation_vec_len']}",
            f"dataset.rotation_parametrization={metadata['rotation_parametrization']}",
            f"dataset.model_path_vec_len={metadata['model_path_vec_len']}",
            "dataset.drake_package_maps=[]",
            "dataset.static_directive=null",
            "dataset.val_ratio=0.1",
            "dataset.test_ratio=0.1",
            # Train and validate on little data.
            "experiment.training.batch_size=1",
            "experiment.validation.batch_size=1",
            "experiment.training.max_epochs=-1",
            "experiment.training.max_steps=2",
            "experiment.validation.val_every_n_step=1",
            # Enable CPU testing.
            "experiment.training.precision=32",
            "experiment.validation.precision=32",
            "experiment.test.precision=32",
            # Disable sampling during validation.
            "algorithm.validation.num_samples_to_render=0",
            "algorithm.validation.num_samples_to_visualize=0",
            "algorithm.validation.num_directives_to_generate=0",
            "algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0",
            # Speed up sampling.
            "algorithm.noise_schedule.num_train_timesteps=2",
            "algorithm.noise_schedule.ddim.num_inference_timesteps=2",
            # Speed up full eval.
            "+num_samples=8",
            "+num_scene_ca_repeats=2",
            "+num_image_ca_repeats=2",
            # Use DDPM.
            "algorithm.trainer=ddpm",
            # Use a small model.
            "algorithm=scene_diffuser_flux_transformer",
            "algorithm.model.hidden_dim=16",
            "algorithm.model.mlp_ratio=1",
            "algorithm.model.num_single_layers=1",
            "algorithm.model.num_double_layers=0",
            "algorithm.model.num_heads=2",
            "algorithm.model.head_dim=8",
            # Enable checkpointing.
            "experiment.training.checkpointing.every_n_train_steps=1",
            "experiment.training.checkpointing.every_n_epochs=null",
            # Enable classifier-free guidance to test conditional sampling.
            "algorithm.classifier_free_guidance.use=True",
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

    def test_evaluate_checkpoints_script(self):
        # Run the main script to save a checkpoint.
        main_cmd = [
            "python",
            "main.py",
            "+name=test",
            *self.fast_args,
        ]
        result = self.run_command(main_cmd)
        self.assertEqual(
            result.returncode, 0, f"Main script failed with stderr: {result.stderr}"
        )

        # Get the checkpoint path from the stdout.
        ckpt_path = self.get_ckpt_path_from_stdout(result.stdout)

        # Run the evaluate_checkpoints script with the checkpoint path.
        script_cmd = [
            "python",
            "scripts/evaluate_checkpoints.py",
            f"load='{ckpt_path}'",
            *self.fast_args,
            "+include_image_metrics=False",
        ]
        result = self.run_command(script_cmd)
        self.assertEqual(
            result.returncode, 0, f"Eval script failed with stderr: {result.stderr}"
        )
        self.assertIn("Running full evaluation on checkpoint", result.stdout)
        self.assertIn("Running unconditional sampling.", result.stdout)
        self.assertNotIn("Calculating embeddings for", result.stdout)

        # Run evaluation with conditional sampling.
        script_cmd = [
            "python",
            "scripts/evaluate_checkpoints.py",
            f"load='{ckpt_path}'",
            *self.fast_args,
            "+conditional=True",
            "+include_image_metrics=False",
        ]
        result = self.run_command(script_cmd)
        self.assertEqual(
            result.returncode, 0, f"Eval script failed with stderr: {result.stderr}"
        )
        self.assertIn("Running full evaluation on checkpoint", result.stdout)
        self.assertIn(
            "Running conditional sampling with the dataset labels.", result.stdout
        )

        # Run evaluation with image metrics.
        script_cmd = [
            "python",
            "scripts/evaluate_checkpoints.py",
            f"load='{ckpt_path}'",
            *self.fast_args,
            "+include_image_metrics=True",
        ]
        result = self.run_command(script_cmd)
        self.assertEqual(
            result.returncode, 0, f"Eval script failed with stderr: {result.stderr}"
        )
        self.assertIn("Running full evaluation on checkpoint", result.stdout)
        self.assertIn("Calculating embeddings for", result.stdout)


if __name__ == "__main__":
    unittest.main()
