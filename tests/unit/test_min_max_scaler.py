import unittest

import numpy as np
import torch

from datasets import Dataset

from steerable_scene_generation.utils.min_max_scaler import (
    MinMaxScaler,
    fit_normalizer,
    fit_normalizer_hf,
)


class TestMinMaxScaler(unittest.TestCase):
    def setUp(self):
        self.scaler = MinMaxScaler()
        self.data = torch.tensor([[-1.0, 2.0], [-0.5, 6.0], [0.0, 18.0], [1.0, 10.0]])

        # Create a batch of scenes with shape (B, N, V).
        self.scenes = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])

    def test_fit(self):
        self.scaler.fit(self.data)
        np.testing.assert_array_almost_equal(
            self.scaler.params["input_stats"]["min"], [-1.0, 2.0]
        )
        np.testing.assert_array_almost_equal(
            self.scaler.params["input_stats"]["max"], [1.0, 18.0]
        )
        np.testing.assert_array_almost_equal(self.scaler.params["scale"], [0.5, 0.0625])
        np.testing.assert_array_almost_equal(self.scaler.params["min"], [0.5, -0.125])

    def test_transform(self):
        self.scaler.fit(self.data)
        transformed_data = self.scaler.transform(self.data)
        expected_transformed_data = np.array(
            [[0.0, 0.0], [0.25, 0.25], [0.5, 1.0], [1.0, 0.5]]
        )
        np.testing.assert_array_almost_equal(
            transformed_data, expected_transformed_data
        )

    def test_transform_for_zero_range(self):
        data = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        self.scaler.fit(data)
        transformed_data = self.scaler.transform(data)
        expected_transformed_data = np.array([[0.0, 0.0], [0.0, 0.0]])
        np.testing.assert_array_almost_equal(
            transformed_data, expected_transformed_data
        )

    def test_inverse_transform(self):
        self.scaler.fit(self.data)
        transformed_data = self.scaler.transform(self.data)
        inverse_transformed_data = self.scaler.inverse_transform(transformed_data)
        np.testing.assert_array_almost_equal(inverse_transformed_data, self.data)

    def test_transform_with_clip(self):
        scaler = MinMaxScaler(clip=True)
        scaler.fit(self.data)
        # First feature is bigger than examples in fitting data.
        transformed_data = scaler.transform(torch.tensor([[2.0, 2.0]]))
        expected_transformed_data = np.array([[1.0, 0.0]])
        np.testing.assert_array_almost_equal(
            transformed_data, expected_transformed_data
        )

    def test_fit_normalizer(self):
        normalizer, state = fit_normalizer(self.scenes)

        # Check that normalizer is properly fitted.
        self.assertTrue(normalizer.is_fitted)
        self.assertEqual(normalizer.output_min, -1.0)
        self.assertEqual(normalizer.output_max, 1.0)
        self.assertTrue(normalizer.clip)

        # Check that min and max values are correct.
        np.testing.assert_array_almost_equal(
            normalizer.params["input_stats"]["min"],
            [1.0, 2.0],
            err_msg="Min values are incorrect.",
        )
        np.testing.assert_array_almost_equal(
            normalizer.params["input_stats"]["max"],
            [7.0, 8.0],
            err_msg="Max values are incorrect.",
        )

        # Check that the state is serializable.
        self.assertIsInstance(state, dict)
        self.assertIn("params", state)
        self.assertIn("is_fitted", state)

        # Test transformation.
        transformed = normalizer.transform(self.scenes.reshape(-1, 2))  # Shape (B*N, V)
        # First element should be mapped to -1, last to 1.
        np.testing.assert_array_almost_equal(
            transformed[0], [-1.0, -1.0], err_msg="First element is incorrect."
        )
        np.testing.assert_array_almost_equal(
            transformed[-1], [1.0, 1.0], err_msg="Last element is incorrect."
        )

    def test_fit_normalizer_hf(self):
        # Create a HF dataset.
        hf_dataset = Dataset.from_dict({"scenes": self.scenes})

        normalizer, state = fit_normalizer_hf(hf_dataset, batch_size=1, num_proc=1)

        # Verify the normalizer is properly fitted.
        self.assertTrue(normalizer.is_fitted)
        self.assertEqual(normalizer.output_min, -1.0)
        self.assertEqual(normalizer.output_max, 1.0)
        self.assertTrue(normalizer.clip)

        # Check that min and max values are correct.
        np.testing.assert_array_almost_equal(
            normalizer.params["input_stats"]["min"],
            [1.0, 2.0],
            err_msg="Min values are incorrect.",
        )
        np.testing.assert_array_almost_equal(
            normalizer.params["input_stats"]["max"],
            [7.0, 8.0],
            err_msg="Max values are incorrect.",
        )

        # Check that the state is serializable.
        self.assertIsInstance(state, dict)
        self.assertIn("params", state)
        self.assertIn("is_fitted", state)

        # Test transformation and inverse transformation.
        transformed = normalizer.transform(self.scenes.reshape(-1, 2))  # Shape (B*N, V)
        # First element should be mapped to -1, last to 1.
        np.testing.assert_array_almost_equal(
            transformed[0], [-1.0, -1.0], err_msg="First element is incorrect."
        )
        np.testing.assert_array_almost_equal(
            transformed[-1], [1.0, 1.0], err_msg="Last element is incorrect."
        )


if __name__ == "__main__":
    unittest.main()
