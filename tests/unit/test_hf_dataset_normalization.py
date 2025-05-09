import unittest

import torch

from datasets import Dataset

from steerable_scene_generation.utils.hf_dataset import (
    normalize_all_scenes,
    unnormalize_all_scenes,
)
from steerable_scene_generation.utils.min_max_scaler import MinMaxScaler


class TestHFDatasetNormalization(unittest.TestCase):
    def setUp(self):
        # Create a mock HF dataset.
        self.data = {
            "scenes": [
                torch.tensor([[-1.0, 2.0], [-0.5, 6.0], [0.0, 18.0], [1.0, 10.0]]),
                torch.tensor([[2.0, 4.0], [3.0, 6.0], [4.0, 8.0], [5.0, 10.0]]),
            ]
        }
        self.hf_dataset = Dataset.from_dict(self.data)

        # Initialize MinMaxScaler and fit on mock data.
        self.scaler = MinMaxScaler()
        flattened_data = torch.cat(self.data["scenes"], dim=0)
        self.scaler.fit(flattened_data)

    def test_normalize_all_scenes(self):
        # Normalize the HF dataset.
        normalized_dataset = normalize_all_scenes(
            self.scaler, self.hf_dataset, batch_size=2
        )

        # Validate normalized values are within [0, 1].
        for batch in normalized_dataset["scenes"]:
            self.assertTrue(torch.all(batch >= 0.0))
            self.assertTrue(torch.all(batch <= 1.0))

    def test_unnormalize_all_scenes(self):
        # Normalize and then unnormalize the HF dataset.
        normalized_dataset = normalize_all_scenes(
            self.scaler, self.hf_dataset, batch_size=2
        )
        unnormalized_dataset = unnormalize_all_scenes(
            self.scaler, normalized_dataset, batch_size=2
        )

        # Validate that unnormalized values match the original dataset.
        for original, unnormalized in zip(
            self.hf_dataset["scenes"], unnormalized_dataset["scenes"]
        ):
            torch.testing.assert_close(original, unnormalized, rtol=1e-5, atol=1e-8)

    def test_normalize_and_clip(self):
        # Test normalization with a fitted scaler and clipping.
        self.scaler.clip = True
        extra_data = {
            "scenes": [
                torch.tensor([[3.0, 20.0], [0.0, -5.0]])
            ]  # Values outside training range
        }
        hf_dataset_with_extra = Dataset.from_dict(extra_data)

        normalized_dataset = normalize_all_scenes(
            self.scaler, hf_dataset_with_extra, batch_size=1
        )
        for batch in normalized_dataset["scenes"]:
            self.assertTrue(torch.all(batch >= 0.0))
            self.assertTrue(torch.all(batch <= 1.0))

    def test_batch_processing(self):
        # Validate that batch processing preserves dataset shape and structure.
        batch_size = 2
        normalized_dataset = normalize_all_scenes(
            self.scaler, self.hf_dataset, batch_size=batch_size
        )
        self.assertEqual(len(normalized_dataset), len(self.hf_dataset))

        for original_batch, normalized_batch in zip(
            self.hf_dataset["scenes"], normalized_dataset["scenes"]
        ):
            self.assertEqual(original_batch.shape, normalized_batch.shape)


if __name__ == "__main__":
    unittest.main()
