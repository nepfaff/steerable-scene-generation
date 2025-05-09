import unittest

import torch

from datasets import Dataset
from omegaconf import OmegaConf
from transformers import BatchEncoding

from steerable_scene_generation.datasets.scene import SceneDataset
from steerable_scene_generation.utils.hf_dataset import (
    get_scene_vec_description_from_metadata,
    load_hf_dataset_with_metadata,
)


class TestSceneDataset(unittest.TestCase):
    def setUp(self):
        # Load configuration from YAML file.
        self.cfg = OmegaConf.load("configurations/dataset/scene.yaml")
        # Manually resolve the classifier-free guidance config.
        base_algo_config = OmegaConf.load(
            "configurations/algorithm/scene_diffuser_base.yaml"
        )
        self.cfg.classifier_free_guidance = base_algo_config.classifier_free_guidance

        # Overwrite dataset path with mock dataset.
        self.cfg.processed_scene_data_path = "tests/datasets/mock_dataset"

        # Load metadata from dataset.
        _, metadata = load_hf_dataset_with_metadata(self.cfg.processed_scene_data_path)

        # Update configuration with metadata.
        self.cfg.max_num_objects_per_scene = metadata["max_num_objects_per_scene"]
        self.cfg.translation_vec_len = metadata["translation_vec_len"]
        self.cfg.rotation_parametrization = metadata["rotation_parametrization"]
        self.cfg.model_path_vec_len = metadata["model_path_vec_len"]

        # Disable subdataset sampling for most tests.
        self.cfg.subdataset_sampling.use = False

        # Set reasonable splits.
        self.cfg.val_ratio = 0.2
        self.cfg.test_ratio = 0.2

        self.scene_vec_desc = get_scene_vec_description_from_metadata(metadata=metadata)
        self.scene_feature_dim = self.scene_vec_desc.get_object_vec_len()

        self.dataset = SceneDataset(cfg=self.cfg, split="training")

    def test_initialization(self):
        self.assertTrue(self.dataset.normalizer.is_fitted)
        self.assertIsNotNone(self.dataset.hf_dataset)

    def test_getitem_without_permutation(self):
        # Test item retrieval without permutation augmentation.
        item = self.dataset[0]
        self.assertIn("scenes", item)
        self.assertIsInstance(item["scenes"], torch.Tensor)

    def test_getitem_with_permutation(self):
        # Temporarily enable permutation augmentation for the test.
        self.dataset.cfg.use_permutation_augmentation = True

        item = self.dataset[0]
        self.assertIn("scenes", item)
        self.assertIsInstance(item["scenes"], torch.Tensor)

        # Reset permutation augmentation
        self.dataset.cfg.use_permutation_augmentation = False

    def test_normalize_scenes(self):
        scenes = torch.rand(2, self.scene_feature_dim)
        normalized_scenes = self.dataset.normalize_scenes(scenes)

        self.assertEqual(normalized_scenes.shape, scenes.shape)
        self.assertTrue(
            torch.all(normalized_scenes >= -1) and torch.all(normalized_scenes <= 1)
        )

    def test_inverse_normalize_scenes(self):
        scenes = (
            torch.rand(2, self.scene_feature_dim) * 2 - 1
        )  # Normalized range [-1, 1]
        unnormalized_scenes = self.dataset.inverse_normalize_scenes(scenes)

        self.assertEqual(unnormalized_scenes.shape, scenes.shape)
        self.assertFalse(
            torch.all(unnormalized_scenes >= -1) and torch.all(unnormalized_scenes <= 1)
        )

    def test_get_sampler(self):
        # Test with custom data batch mix disabled.
        self.dataset.cfg.custom_data_batch_mix.use = False
        sampler = self.dataset.get_sampler()
        self.assertIsNone(sampler)

        # Test with custom data batch mix enabled.
        self.dataset.cfg.custom_data_batch_mix.use = True

        # Dynamically generate labels based on dataset length.
        dataset_length = len(self.dataset.hf_dataset)
        labels = [i % 2 for i in range(dataset_length)]  # Alternate labels (0, 1)
        self.dataset.hf_dataset = self.dataset.hf_dataset.add_column("labels", labels)

        sampler = self.dataset.get_sampler()
        self.assertIsInstance(sampler, torch.utils.data.WeightedRandomSampler)

    def test_len(self):
        # Check length of dataset.
        self.assertEqual(len(self.dataset), len(self.dataset.hf_dataset))

    def test_get_and_set_data(self):
        # Test get data.
        data = self.dataset.get_all_data(normalized=False)
        self.assertIsInstance(data, dict)
        self.assertIn("scenes", data)
        self.assertIsInstance(data["scenes"], torch.Tensor)

        # Remove one element.
        for key in data:
            data[key] = data[key][:-1]

        # Test set data.
        self.dataset.set_data(data, normalized=False)
        self.assertEqual(len(self.dataset), len(data["scenes"]))

        # Test get data with indices.
        indices_data = self.dataset.get_all_data(
            scene_indices=torch.tensor([0, 2]), normalized=False
        )
        self.assertIsInstance(indices_data, dict)
        self.assertEqual(len(indices_data["scenes"]), 2)
        # Check that got the correct scenes.
        self.assertTrue(torch.all(data["scenes"][0] == indices_data["scenes"][0]))
        self.assertTrue(torch.all(data["scenes"][2] == indices_data["scenes"][1]))

    def test_get_all_data_with_label_filtering(self):
        # Add labels to dataset.
        labels = torch.tensor([0, 1, 0, 1, 0])
        scenes = torch.rand(len(labels), 10, self.scene_feature_dim)
        new_data = {"scenes": scenes.tolist(), "labels": labels.tolist()}
        self.dataset.hf_dataset = Dataset.from_dict(new_data)
        self.dataset.hf_dataset.set_format("torch")

        # Fetch data for a specific label.
        filtered_data = self.dataset.get_all_data(label=0)
        self.assertEqual(len(filtered_data["scenes"]), (labels == 0).sum().item())
        self.assertTrue(torch.all(filtered_data["labels"] == 0))

    def test_get_all_data_with_scene_indices(self):
        # Fetch specific scenes by index.
        scenes = torch.rand(5, 10, self.scene_feature_dim)
        labels = torch.tensor([0, 1, 0, 1, 0])
        new_data = {"scenes": scenes.tolist(), "labels": labels.tolist()}
        self.dataset.hf_dataset = Dataset.from_dict(new_data)
        self.dataset.hf_dataset.set_format("torch")

        indices = torch.tensor([1, 3])
        selected_data = self.dataset.get_all_data(scene_indices=indices)
        self.assertEqual(len(selected_data["scenes"]), len(indices))
        self.assertTrue(torch.all(selected_data["scenes"] == scenes[indices]))
        self.assertTrue(torch.all(selected_data["labels"] == labels[indices]))

    def test_add_classifier_free_guidance_uncond_data(self):
        # Enable classifier-free guidance.
        self.dataset.cfg.classifier_free_guidance.use = True
        scenes = torch.rand(3, 10, self.scene_feature_dim)  # Mock scene data.
        text_cond = self.dataset.tokenizer(["label1", "label2", "label3"])
        data = {"scenes": scenes, "text_cond": text_cond}

        # Add unconditioned data.
        updated_data = self.dataset.add_classifier_free_guidance_uncond_data(data)
        self.assertIn("text_cond", updated_data)
        self.assertEqual(
            updated_data["text_cond"]["input_ids"].shape[0], 6
        )  # Original + uncond.
        self.assertEqual(updated_data["text_cond"]["attention_mask"].shape[0], 6)
        # Check that the original text condition is preserved.
        self.assertTrue(
            torch.all(
                text_cond["input_ids"] == updated_data["text_cond"]["input_ids"][:3]
            )
        )
        self.assertTrue(
            torch.all(
                text_cond["attention_mask"]
                == updated_data["text_cond"]["attention_mask"][:3]
            )
        )

    def test_replace_cond_data(self):
        # Mock scenes and text conditions.
        scenes = torch.rand(3, 10, self.scene_feature_dim)  # Mock scene data.
        text_cond = {
            "input_ids": torch.randint(0, 1000, (3, 5)),
            "attention_mask": torch.ones(3, 5),
        }
        data = {"scenes": scenes, "text_cond": text_cond}

        # Replace conditioning data.
        new_labels = ["label1", "label2", "label3"]
        updated_data = self.dataset.replace_cond_data(data, new_labels)
        self.assertIn("text_cond", updated_data)
        self.assertEqual(len(updated_data["text_cond"]["input_ids"]), len(new_labels))

    def test_sample_cond_data(self):
        # Mock data with BatchEncoding and torch.Tensor.
        scenes = torch.rand(5, 10, self.scene_feature_dim)
        labels = torch.tensor([0, 1, 0, 1, 0])
        text_cond = BatchEncoding(
            {
                "input_ids": torch.randint(0, 1000, (5, 5)),
                "attention_mask": torch.ones(5, 5),
            }
        )
        data = {"scenes": scenes, "labels": labels, "text_cond": text_cond}

        # Sample without replacement.
        sampled_data = self.dataset.sample_data_dict(data, num_items=3)
        self.assertEqual(len(sampled_data["scenes"]), 3)
        self.assertEqual(len(sampled_data["labels"]), 3)
        self.assertEqual(sampled_data["text_cond"]["input_ids"].shape[0], 3)

        # Sample with replacement.
        sampled_data_with_replacement = self.dataset.sample_data_dict(data, num_items=7)
        self.assertEqual(len(sampled_data_with_replacement["scenes"]), 7)
        self.assertEqual(len(sampled_data_with_replacement["labels"]), 7)
        self.assertEqual(
            sampled_data_with_replacement["text_cond"]["input_ids"].shape[0], 7
        )

        self.assertIsInstance(sampled_data_with_replacement["scenes"], torch.Tensor)
        self.assertIsInstance(sampled_data_with_replacement["labels"], torch.Tensor)
        self.assertIsInstance(sampled_data_with_replacement["text_cond"], BatchEncoding)

    def _setup_mock_subdatasets(self, with_unique_identifiers=False) -> None:
        """Helper method to set up mock subdatasets for testing.

        Args:
            with_unique_identifiers: If True, adds unique values to each subdataset
                to make them distinguishable.
        """
        # Create mock subdataset metadata.
        subdataset_names = ["dataset1", "dataset2", "dataset3"]
        subdataset_ranges = [(0, 10), (10, 20), (20, 30)]

        # Create a new dataset with mock subdataset metadata.
        if with_unique_identifiers:
            # Create scenes with unique identifiers for each subdataset.
            scenes = torch.zeros(30, 10, self.scene_feature_dim)
            # Set a unique identifier for each subdataset in the first element.
            scenes[0:10, 0, 0] = 1.0  # dataset1 identifier
            scenes[10:20, 0, 0] = 2.0  # dataset2 identifier
            scenes[20:30, 0, 0] = 3.0  # dataset3 identifier
        else:
            # Create random scenes.
            scenes = torch.rand(30, 10, self.scene_feature_dim)

        # Create mock data with all required fields.
        new_data = {
            "scenes": scenes.tolist(),
            "language_annotation": ["mock description"] * 30,
        }

        self.dataset.hf_dataset = Dataset.from_dict(new_data)
        self.dataset.hf_dataset.set_format("torch")
        self.dataset.subdataset_names = subdataset_names
        self.dataset.subdataset_ranges = subdataset_ranges

    def test_subdataset_sampling_setup(self):
        """Test the setup of subdataset sampling."""
        self._setup_mock_subdatasets()

        # Configure subdataset sampling.
        self.dataset.cfg.subdataset_sampling.use = True
        self.dataset.cfg.subdataset_sampling.probabilities = {
            "dataset1": 0.2,
            "dataset2": 0.3,
            "dataset3": 0.5,
        }

        # Setup subdataset sampling.
        self.dataset.setup_subdataset_sampling(
            self.dataset.cfg.subdataset_sampling, hf_dataset=self.dataset.hf_dataset
        )

        # Check that cumulative probabilities are set up correctly.
        self.assertEqual(len(self.dataset.subdataset_cum_probs), 3)
        self.assertAlmostEqual(self.dataset.subdataset_cum_probs[-1], 1.0)

        # Test with invalid probabilities (missing dataset).
        invalid_probs = {
            "dataset1": 0.2,
            "dataset2": 0.3,
            # Missing dataset3
        }
        self.dataset.cfg.subdataset_sampling.probabilities = invalid_probs
        with self.assertRaises(ValueError):
            self.dataset.setup_subdataset_sampling(
                self.dataset.cfg.subdataset_sampling, hf_dataset=self.dataset.hf_dataset
            )

        # Test with invalid probabilities (sum != 1).
        invalid_probs = {
            "dataset1": 0.2,
            "dataset2": 0.3,
            "dataset3": 0.3,  # Sum = 0.8
        }
        self.dataset.cfg.subdataset_sampling.probabilities = invalid_probs
        with self.assertRaises(ValueError):
            self.dataset.setup_subdataset_sampling(
                self.dataset.cfg.subdataset_sampling, hf_dataset=self.dataset.hf_dataset
            )

        # Reset.
        self.dataset.cfg.subdataset_sampling.use = False

    def test_sample_subdataset_index(self):
        """Test sampling a subdataset index based on probabilities."""
        self._setup_mock_subdatasets()

        # Configure subdataset sampling with extreme probabilities for deterministic
        # testing.
        self.dataset.cfg.subdataset_sampling.use = True
        self.dataset.cfg.subdataset_sampling.probabilities = {
            "dataset1": 1.0,  # Always select dataset1
            "dataset2": 0.0,
            "dataset3": 0.0,
        }

        # Setup subdataset sampling.
        self.dataset.use_subdataset_sampling = True
        self.dataset.setup_subdataset_sampling(
            self.dataset.cfg.subdataset_sampling, hf_dataset=self.dataset.hf_dataset
        )

        # Sample subdataset index (should always be 0 with our probabilities).
        sampled_idx = self.dataset.sample_subdataset_index()
        self.assertEqual(sampled_idx, 0)

        # Test with different probabilities.
        self.dataset.cfg.subdataset_sampling.probabilities = {
            "dataset1": 0.0,
            "dataset2": 0.0,
            "dataset3": 1.0,  # Always select dataset3
        }
        self.dataset.setup_subdataset_sampling(
            self.dataset.cfg.subdataset_sampling, hf_dataset=self.dataset.hf_dataset
        )
        sampled_idx = self.dataset.sample_subdataset_index()
        self.assertEqual(sampled_idx, 2)

        # Test when subdataset sampling is not enabled.
        self.dataset.use_subdataset_sampling = False
        with self.assertRaises(RuntimeError):
            self.dataset.sample_subdataset_index()

        # Reset.
        self.dataset.cfg.subdataset_sampling.use = False

    def test_sample_from_subdataset(self):
        """Test sampling an item from a specific subdataset."""
        self._setup_mock_subdatasets()

        # Setup subdataset sampling.
        self.dataset.cfg.subdataset_sampling.use = True
        self.dataset.cfg.subdataset_sampling.probabilities = {
            "dataset1": 0.33,
            "dataset2": 0.33,
            "dataset3": 0.34,
        }
        self.dataset.use_subdataset_sampling = True
        self.dataset.setup_subdataset_sampling(
            self.dataset.cfg.subdataset_sampling, hf_dataset=self.dataset.hf_dataset
        )

        # Sample from subdataset 0 (should be in range 0-9).
        idx = self.dataset.sample_from_subdataset(0)
        self.assertTrue(0 <= idx < 10)

        # Sample from subdataset 1 (should be in range 10-19).
        idx = self.dataset.sample_from_subdataset(1)
        self.assertTrue(10 <= idx < 20)

        # Sample from subdataset 2 (should be in range 20-29).
        idx = self.dataset.sample_from_subdataset(2)
        self.assertTrue(20 <= idx < 30)

        # Reset.
        self.dataset.cfg.subdataset_sampling.use = False
        self.dataset.use_subdataset_sampling = False

    def test_getitem_with_subdataset_sampling(self):
        """Test __getitem__ with subdataset sampling enabled."""
        self._setup_mock_subdatasets(with_unique_identifiers=True)

        # Configure subdataset sampling with extreme probabilities for deterministic testing
        self.dataset.cfg.subdataset_sampling.use = True
        self.dataset.cfg.subdataset_sampling.probabilities = {
            "dataset1": 1.0,  # Always select dataset1
            "dataset2": 0.0,
            "dataset3": 0.0,
        }

        # Setup subdataset sampling.
        self.dataset.use_subdataset_sampling = True
        self.dataset.setup_subdataset_sampling(
            self.dataset.cfg.subdataset_sampling, hf_dataset=self.dataset.hf_dataset
        )

        # Get an item (should be from dataset1).
        item = self.dataset[0]  # Index is ignored with subdataset sampling
        self.assertEqual(item["scenes"][0, 0].item(), 1.0)

        # Change probabilities to always select dataset2.
        self.dataset.cfg.subdataset_sampling.probabilities = {
            "dataset1": 0.0,
            "dataset2": 1.0,  # Always select dataset2
            "dataset3": 0.0,
        }
        self.dataset.setup_subdataset_sampling(
            self.dataset.cfg.subdataset_sampling, hf_dataset=self.dataset.hf_dataset
        )

        # Get an item (should be from dataset2).
        item = self.dataset[0]  # Index is ignored with subdataset sampling
        self.assertEqual(item["scenes"][0, 0].item(), 2.0)

        # Disable subdataset sampling.
        self.dataset.use_subdataset_sampling = False

        # Get a specific item (should respect the index).
        item = self.dataset[5]
        self.assertEqual(item["scenes"][0, 0].item(), 1.0)  # From dataset1

        item = self.dataset[15]
        self.assertEqual(item["scenes"][0, 0].item(), 2.0)  # From dataset2

        item = self.dataset[25]
        self.assertEqual(item["scenes"][0, 0].item(), 3.0)  # From dataset3

        # Reset.
        self.dataset.cfg.subdataset_sampling.use = False


if __name__ == "__main__":
    unittest.main()
