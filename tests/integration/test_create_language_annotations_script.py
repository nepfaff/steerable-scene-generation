import os
import subprocess
import tempfile
import unittest

from typing import List

from steerable_scene_generation.utils.hf_dataset import load_hf_dataset_with_metadata


class TestCreateLanguageAnnotationsScript(unittest.TestCase):
    def setUp(self):
        self.mock_dataset_path = "tests/datasets/mock_dataset"

    def run_command(self, command: List[str]) -> subprocess.CompletedProcess:
        # Run the command as a subprocess
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return result

    def test_all_language_modes(self):
        # Create a temporary directory for the output dataset.
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test each language mode.
            language_modes = {
                "object_names": {
                    "args": ["--subset_probability", "0.5"],
                    "verifications": lambda x: "box" in x.lower()
                    or "sphere" in x.lower()
                    or "cylinder" in x.lower(),
                },
                "spatial_relations": {
                    "args": [
                        "--num_spatial_relation_annotations",
                        "2",
                        "--max_spatial_relationships",
                        "3",
                    ],
                    "verifications": lambda x: any(
                        kw in x.lower()
                        for kw in [
                            "left",
                            "right",
                            "above",
                            "below",
                            "front",
                            "behind",
                            "next to",
                        ]
                    ),
                },
                "object_number": {
                    "args": [],
                    "verifications": lambda x: "a scene with" in x.lower()
                    and any(num in x for num in "0123456789")
                    and "object" in x.lower(),
                },
            }

            for mode, config in language_modes.items():
                output_path = os.path.join(temp_dir, f"annotated_dataset_{mode}")

                # Build command for this mode.
                script_cmd = [
                    "python",
                    "scripts/create_language_annotations.py",
                    self.mock_dataset_path,
                    output_path,
                    "--scene_type",
                    "table_top",
                    "--language_mode",
                    mode,
                    "--max_workers",
                    "1",
                    "--spatial_relation_distance_threshold",
                    "100.0",
                    *config["args"],
                ]

                # Run the script.
                result = self.run_command(script_cmd)
                self.assertEqual(
                    result.returncode,
                    0,
                    f"Script failed for mode {mode} with stderr: {result.stderr}",
                )

                # Load and verify the annotated dataset.
                annotated_dataset, _ = load_hf_dataset_with_metadata(output_path)

                # Basic checks.
                self.assertIn(
                    "language_annotation",
                    annotated_dataset.features,
                    f"Language annotation missing for mode {mode}",
                )
                self.assertTrue(
                    len(annotated_dataset) > 0, f"Empty dataset for mode {mode}"
                )

                # Mode-specific verification.
                for annotation in annotated_dataset["language_annotation"]:
                    self.assertIsInstance(
                        annotation, str, f"Non-string annotation found in mode {mode}"
                    )
                    self.assertTrue(
                        config["verifications"](annotation),
                        f"Invalid annotation format in mode {mode}: {annotation}",
                    )

                # Special checks for spatial_relations mode.
                if mode == "spatial_relations":
                    original_dataset, _ = load_hf_dataset_with_metadata(
                        self.mock_dataset_path
                    )
                    self.assertGreater(
                        len(annotated_dataset),
                        len(original_dataset),
                        "Spatial relations mode should create more than 1 annotation "
                        "per scene",
                    )
                    self.assertLessEqual(
                        len(annotated_dataset),
                        len(original_dataset) * 2,
                        "Spatial relations mode should create at most 2 annotations "
                        "per scene",
                    )


if __name__ == "__main__":
    unittest.main()
