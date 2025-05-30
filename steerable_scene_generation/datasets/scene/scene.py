import json
import logging
import os
import random

from functools import cache
from typing import Any, Union

import torch

from datasets import Dataset
from omegaconf import DictConfig
from torch.utils.data import WeightedRandomSampler
from transformers import BatchEncoding

from steerable_scene_generation.algorithms.common.txt_encoding import (
    concat_batch_encodings,
    load_txt_encoder_from_config,
)
from steerable_scene_generation.datasets.common import BaseDataset
from steerable_scene_generation.datasets.common.shuffled_streaming_dataset import (
    InfiniteShuffledStreamingDataset,
)
from steerable_scene_generation.utils.hf_dataset import load_hf_dataset_with_metadata
from steerable_scene_generation.utils.min_max_scaler import MinMaxScaler

console_logger = logging.getLogger(__name__)


class SceneDataset(BaseDataset):
    def __init__(self, cfg: DictConfig, split: str, ckpt_path: str | None = None):
        """
        Args:
            cfg: a DictConfig object defined by `configurations/dataset/scene.yaml`.
            split: One of "training", "validation", "test".
            ckpt_path: The optional checkpoint path.
        """
        self.cfg = cfg

        # Check if the path is a Hugging Face Hub dataset ID.
        is_hub_dataset = "/" in cfg.processed_scene_data_path and not os.path.exists(
            cfg.processed_scene_data_path
        )

        if is_hub_dataset or os.path.isdir(cfg.processed_scene_data_path):
            # Load the dataset.
            hf_dataset, metadata = load_hf_dataset_with_metadata(
                cfg.processed_scene_data_path, keep_in_memory=cfg.keep_dataset_in_memory
            )
            hf_dataset.set_format("torch")

            hf_dataset = self._subsample_dataset_if_enabled(hf_dataset)

            self._validate_dataset_structure(hf_dataset, metadata)

            # Store info for subdataset sampling.
            self.subdataset_ranges = metadata.get("subdataset_ranges", None)
            self.subdataset_names = metadata.get("subdataset_names", None)

            self.use_subdataset_sampling = self.cfg.subdataset_sampling.use
            self._validate_subdataset_config()

            self.hf_dataset = self._perform_train_test_validation_split(
                hf_dataset=hf_dataset, split=split
            )

            self._validate_dataset_structure(hf_dataset, metadata)
        else:
            # Load the dataset metadata.
            metadata_path = os.path.expanduser(cfg.processed_scene_data_path)
            with open(metadata_path, "r") as metadata_file:
                metadata = json.load(metadata_file)

            self.use_subdataset_sampling = False
            self.hf_dataset = None
            logging.warning("No dataset path provided. Only sampling is supported.")

        self.normalizer = self._setup_normalizer(ckpt_path, metadata)

        self._setup_tokenizers()

    def _subsample_dataset_if_enabled(self, hf_dataset: Dataset) -> Dataset:
        """
        Subsamples the dataset if the random dataset sampling is enabled.
        """
        if self.cfg.random_dataset_sampling.use and self.cfg.subdataset_sampling.use:
            raise NotImplementedError(
                "Cannot use both subdataset sampling and random dataset sampling!"
            )

        if self.cfg.random_dataset_sampling.use:
            num_samples = int(self.cfg.random_dataset_sampling.num_samples)
            if len(hf_dataset) < num_samples:
                raise ValueError(
                    f"Dataset size ({len(hf_dataset)}) is smaller than the number "
                    f"of samples to sample ({num_samples})!"
                )
            hf_dataset = hf_dataset.select(
                torch.randperm(len(hf_dataset))[:num_samples]
            )
            console_logger.info(
                f"Using random dataset sampling with {num_samples} samples."
            )

        return hf_dataset

    def _perform_train_test_validation_split(
        self, hf_dataset: Dataset, split: str
    ) -> Dataset:
        """
        Performs a train-test split on the dataset.
        """
        if self.use_subdataset_sampling:
            # Don't split the dataset as it would mess up the subdataset sampling
            # indices due to random shuffling.
            self.setup_subdataset_sampling(self.cfg.subdataset_sampling, hf_dataset)

            # Calculate what the split size would be to maintain consistent dataset
            # sizes.
            total_size = len(hf_dataset)
            if split == "training":
                split_size = int(
                    total_size * (1.0 - self.cfg.val_ratio - self.cfg.test_ratio)
                )
            elif split == "validation":
                split_size = int(total_size * self.cfg.val_ratio)
            elif split == "test":
                split_size = int(total_size * self.cfg.test_ratio)
            else:
                raise ValueError(f"Invalid split: {split}")

            self.sampling_dataset_length = split_size
            console_logger.info(
                f"Using subdataset sampling with {split} split "
                f"length: {self.sampling_dataset_length}"
            )
            return hf_dataset

        # Only perform train-test split when not using subdataset sampling.
        train_ratio = (
            1.0 - self.cfg.val_ratio - self.cfg.test_ratio
            if len(hf_dataset) > 1
            else 1.0
        )
        train_testval_split = hf_dataset.train_test_split(train_size=train_ratio)
        if split == "training":
            return train_testval_split["train"]
        else:
            # Split further into validation and test.
            val_test_split = train_testval_split["test"].train_test_split(
                train_size=self.cfg.val_ratio
                / (self.cfg.val_ratio + self.cfg.test_ratio)
            )
            if split == "validation":
                return val_test_split["train"]
            elif split == "test":
                return val_test_split["test"]
            else:
                raise ValueError(f"Invalid split: {split}")

    def _setup_normalizer(
        self, ckpt_path: str | None, metadata: dict[str, Any]
    ) -> MinMaxScaler:
        """
        Sets up the normalizer.
        """
        normalizer = MinMaxScaler(output_min=-1.0, output_max=1.0, clip=True)

        # Check if the checkpoint contains the normalizer.
        ckpt_normalizer_state = None
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if "normalizer_state" in ckpt:
                ckpt_normalizer_state = ckpt["normalizer_state"]

                # Validate the normalizer state.
                if self.hf_dataset is not None:
                    scene_shape = self.hf_dataset[0]["scenes"].shape
                    normalizer_scene_vec_len = len(
                        ckpt["normalizer_state"]["params"]["scale"]
                    )
                    if normalizer_scene_vec_len != scene_shape[-1]:
                        raise ValueError(
                            "Normalizer scene vector length "
                            f"({normalizer_scene_vec_len}) does not match the dataset "
                            f"scene vector length ({scene_shape[-1]})."
                        )

        # Set the normalizer state.
        # Note that the dataset scenes are assumed to be already normalized.
        if ckpt_normalizer_state is None:
            normalizer.load_serializable_state(metadata["normalizer_state"])
            console_logger.info("Loaded normalizer state from the dataset.")
        else:
            normalizer.load_state(ckpt_normalizer_state)
            console_logger.info("Loaded normalizer state from the checkpoint.")
        if not normalizer.is_fitted:
            raise ValueError("Normalizer is not fitted!")

        return normalizer

    def _setup_tokenizers(self):
        """
        Sets up the tokenizers.
        """
        # Load the tokenizer if using classifier-free guidance.
        self.tokenizer, self.tokenizer_coarse = None, None
        if not self.cfg.classifier_free_guidance.use:
            return

        # Setup the primary tokenizer.
        self.masking_prop = self.cfg.classifier_free_guidance.masking_prob
        self.tokenizer, _ = load_txt_encoder_from_config(
            self.cfg, component="tokenizer"
        )

        # Setup the coarse tokenizer if configured.
        use_coarse = self.cfg.classifier_free_guidance.txt_encoder_coarse is not None
        if use_coarse:
            self.masking_prop_coarse = (
                self.cfg.classifier_free_guidance.masking_prob_coarse
            )
            self.tokenizer_coarse, _ = load_txt_encoder_from_config(
                self.cfg, is_coarse=True, component="tokenizer"
            )

        # Pre-cache empty encodings for masked prompts.
        self._setup_tokenization_caches()

        # Setup static prompt caching if configured.
        if self.cfg.static_subdataset_prompts.use and hasattr(self, "subdataset_names"):
            self._setup_static_prompt_caches()

    def _setup_tokenization_caches(self):
        """
        Sets up caches for tokenization to improve performance.
        This includes caching empty string tokenization for masked prompts.
        """
        # Pre-generate empty token encodings for masked prompts.
        if self.tokenizer is not None:
            self._empty_encoding = self.tokenizer([""])
            self._empty_encoding = BatchEncoding(
                {k: v.squeeze(0) for k, v in self._empty_encoding.items()}
            )
            console_logger.info("Pre-cached empty encoding for regular tokenizer.")

        if self.tokenizer_coarse is not None:
            self._empty_encoding_coarse = self.tokenizer_coarse([""])
            self._empty_encoding_coarse = BatchEncoding(
                {k: v.squeeze(0) for k, v in self._empty_encoding_coarse.items()}
            )
            console_logger.info("Pre-cached empty encoding for coarse tokenizer.")

    def _setup_static_prompt_caches(self):
        """
        Pre-caches tokenized versions of static subdataset prompts.
        """
        if self.tokenizer is None and self.tokenizer_coarse is None:
            return

        self._tokenized_prompts_cache = {}
        for prompt in self.cfg.static_subdataset_prompts.name_to_prompt.values():
            self._tokenized_prompts_cache[prompt] = {}

            if self.tokenizer is not None:
                text_cond = self.tokenizer([prompt])
                self._tokenized_prompts_cache[prompt]["regular"] = BatchEncoding(
                    {k: v.squeeze(0) for k, v in text_cond.items()}
                )

            if self.tokenizer_coarse is not None:
                text_cond_coarse = self.tokenizer_coarse([prompt])
                self._tokenized_prompts_cache[prompt]["coarse"] = BatchEncoding(
                    {k: v.squeeze(0) for k, v in text_cond_coarse.items()}
                )

        console_logger.info(
            f"Pre-cached tokenization for {len(self._tokenized_prompts_cache)} static "
            "prompts."
        )

    def normalize_scenes(self, scenes: torch.Tensor) -> torch.Tensor:
        """
        Normalize scenes to [-1, 1].

        Args:
            scenes: Scenes to normalize. Shape (B, N, O) where B is the number of scenes,
                N is the number of objects, and O is the object feature vector length.

        Returns:
            torch.Tensor: Normalized scenes.
        """
        normalized_scenes = self.normalizer.transform(
            scenes.reshape(-1, scenes.shape[-1])
        ).reshape(scenes.shape)
        return normalized_scenes

    def inverse_normalize_scenes(self, scenes: torch.Tensor) -> torch.Tensor:
        """
        Inverse normalize scenes from [-1, 1].

        Args:
            scenes: Scenes to inverse normalize. Shape (B, N, O) where B is the number of
                scenes, N is the number of objects, and O is the object feature vector
                length.

        Returns:
            torch.Tensor: Inverse normalized scenes.
        """
        unormalized_scenes = self.normalizer.inverse_transform(
            scenes.reshape(-1, scenes.shape[-1])
        ).reshape(scenes.shape)

        return unormalized_scenes

    def __len__(self) -> int:
        """
        Returns the length of the dataset. When subdataset sampling is enabled,
        returns a fixed length that can be configured.
        """
        if self.hf_dataset is None:
            return 0

        if self.use_subdataset_sampling:
            return self.sampling_dataset_length
        return len(self.hf_dataset)

    def _get_item(self, idx: int) -> dict[str, torch.Tensor]:
        if self.hf_dataset is None:
            raise ValueError("Dataset is not loaded!")

        if not self.use_subdataset_sampling:
            return self.hf_dataset[idx]

        # If subdataset sampling is enabled, override the index.
        subdataset_idx = self.sample_subdataset_index()
        if self.use_infinite_iterators:
            return next(self.subdataset_iterator_objects[subdataset_idx])
        idx = self.sample_from_subdataset(subdataset_idx)
        return self.hf_dataset[idx]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self._get_item(idx)

        if self.cfg.use_permutation_augmentation:
            scene = item["scenes"]  # Shape (N, O)
            perm = torch.randperm(len(scene))
            item["scenes"] = scene[perm]

        if self.tokenizer is not None or self.tokenizer_coarse is not None:
            if self.cfg.static_subdataset_prompts.use:
                subdataset_name = self.get_subdataset_name_from_index(idx)
                prompt = self.cfg.static_subdataset_prompts.name_to_prompt[
                    subdataset_name
                ]

                # Use cached tokenization for static prompts.
                if hasattr(self, "_tokenized_prompts_cache"):
                    if self.tokenizer is not None:
                        if random.random() >= self.masking_prop:
                            item["text_cond"] = self._tokenized_prompts_cache[prompt][
                                "regular"
                            ]
                        else:
                            item["text_cond"] = self._empty_encoding

                    if self.tokenizer_coarse is not None:
                        if random.random() >= self.masking_prop_coarse:
                            item["text_cond_coarse"] = self._tokenized_prompts_cache[
                                prompt
                            ]["coarse"]
                        else:
                            item["text_cond_coarse"] = self._empty_encoding_coarse

                    return item
            else:
                prompt = item["language_annotation"]

            if self.tokenizer is not None:
                if self.masking_prop < random.random():
                    text_cond = self.tokenizer(prompt)
                else:
                    text_cond = (
                        self._empty_encoding
                        if hasattr(self, "_empty_encoding")
                        else self.tokenizer([""])
                    )

                # Drop the batch dimensions.
                item["text_cond"] = BatchEncoding(
                    {k: v.squeeze(0) for k, v in text_cond.items()}
                )

            if self.tokenizer_coarse is not None:
                if self.masking_prop_coarse < random.random():
                    text_cond_coarse = self.tokenizer_coarse(prompt)
                else:
                    text_cond_coarse = (
                        self._empty_encoding_coarse
                        if hasattr(self, "_empty_encoding_coarse")
                        else self.tokenizer_coarse([""])
                    )

                # Drop the batch dimensions.
                item["text_cond_coarse"] = BatchEncoding(
                    {k: v.squeeze(0) for k, v in text_cond_coarse.items()}
                )

        return item

    def add_classifier_free_guidance_uncond_data(
        self, data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Adds classifier-free guidance unconditioned data to the input data. The
        resulting data will contain the original data along with the classifier-free
        guidance unconditioned data, concatenated such that the original data comes
        first.

        Args:
            data (dict[str, Any]): The input data.

        Returns:
            dict[str, Any]: The data with added classifier-free guidance unconditioned
            data.
        """
        if not self.cfg.classifier_free_guidance.use:
            return data

        if not data["scenes"].dim() == 3:
            raise NotImplementedError("Only batched data is supported right now.")

        uncond_txt = [""] * len(data["scenes"])

        if "language_annotation" in data:
            data["language_annotation"] = data["language_annotation"] + uncond_txt

        if "text_cond" in data:
            device = data["text_cond"]["input_ids"].device
            uncond_cond = self.tokenizer(uncond_txt).to(device)
            data["text_cond"] = concat_batch_encodings([data["text_cond"], uncond_cond])

        if "text_cond_coarse" in data:
            device = data["text_cond_coarse"]["input_ids"].device
            uncond_cond_coarse = self.tokenizer_coarse(uncond_txt).to(device)
            data["text_cond_coarse"] = concat_batch_encodings(
                [data["text_cond_coarse"], uncond_cond_coarse]
            )

        return data

    def replace_cond_data(
        self, data: dict[str, Any], txt_labels: str | list[str]
    ) -> dict[str, Any]:
        """
        Replaces the conditioning data in the input data with the provided text labels.

        Args:
            data (dict[str, Any]): The input data.
            txt_labels (str | list[str]): The text labels to use for conditioning.

        Returns:
            dict[str, Any]: The data with replaced conditioning data.
        """
        if isinstance(txt_labels, str):
            txt_labels = [txt_labels] * len(data["scenes"])

        if len(txt_labels) != len(data["scenes"]):
            raise ValueError(
                "The number of text labels does not match the number of scenes."
            )

        if "text_cond" in data and self.tokenizer is not None:
            data["text_cond"] = self.tokenizer(txt_labels)

        if "text_cond_coarse" in data and self.tokenizer_coarse is not None:
            data["text_cond_coarse"] = self.tokenizer_coarse(txt_labels)

        return data

    @staticmethod
    def sample_data_dict(data: dict[str, Any], num_items: int) -> dict[str, Any]:
        """
        Sample `num_items` from `data`. Sample with replacement if `data` contains less
        than `num_items` items.

        Args:
            data (dict[str, Any]): The data to sample.

        Returns:
            dict[str, Any]: The sampled data of length `num_items`.
        """
        total_items = len(data["scenes"])

        if num_items <= total_items:
            # Sample without replacement.
            sample_indices = torch.randperm(total_items)[:num_items]
        else:
            # Sample with replacement.
            sample_indices = torch.randint(0, total_items, (num_items,))

        # Create the sampled data dictionary.
        sampled_data = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                sampled_data[key] = value[sample_indices]
            elif isinstance(value, list):
                sampled_data[key] = [value[i] for i in sample_indices]
            elif isinstance(value, BatchEncoding):
                sampled_data[key] = BatchEncoding(
                    {k: v[sample_indices] for k, v in value.items()}
                )
            else:
                raise ValueError(
                    f"Unsupported data type '{type(value)}' for key '{key}'"
                )

        return sampled_data

    def get_sampler(self) -> Union[WeightedRandomSampler, None]:
        """
        Returns a sampler for weighted random sampling of the dataset based on the
        dataset labels.
        This is an alternative to the two-step sampling process used in
        `sample_subdataset_index` and `sample_from_subdataset`.
        """
        if not self.cfg.custom_data_batch_mix.use:
            return None

        if "labels" not in self.hf_dataset.column_names:
            raise ValueError("Dataset does not contain labels!")

        labels = self.hf_dataset["labels"]

        # Calculate the number of samples for each class.
        class_counts = torch.bincount(labels)

        # Calculate weights for each class.
        class_weights = [
            p / count if count > 0 else 0.0
            for p, count in zip(
                self.cfg.custom_data_batch_mix.label_probs, class_counts
            )
        ]

        # Create a list of weights for each sample in the dataset.
        weights = [class_weights[label] for label in labels]

        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(self),
            replacement=True,
        )
        return sampler

    @cache
    def get_all_data(
        self,
        normalized: bool = True,
        label: int = None,
        scene_indices: torch.Tensor | None = None,
        only_scenes: bool = False,
    ) -> dict[str, Any]:
        """
        Returns all data in the dataset, including scenes and additional attributes.

        Args:
            normalized (bool, optional): Whether to return normalized scenes.
            label (int, optional): If not None, only data that correspond to that
                label are returned. This option is ignored if the dataset does not
                contain labels.
            scene_indices (torch.Tensor, optional): If not None, only data at the
                specified indices are returned.
            only_scenes (bool, optional): If True, only the scenes are returned.

        Returns:
            dict[str, Any]: All data in the dataset, including "scenes" and
                additional attributes.
        """
        if self.hf_dataset is None:
            raise ValueError("Dataset is not loaded!")

        if only_scenes and label is not None:
            raise ValueError("Cannot specify both 'only_scenes' and 'label'.")

        # Use indexing to fetch only the required data.
        if scene_indices is not None:
            scene_indices = scene_indices.tolist()  # Convert to list for indexing
            raw_data = self.hf_dataset.select(scene_indices)
        else:
            raw_data = self.hf_dataset

        raw_scenes = raw_data["scenes"]
        if only_scenes:
            return {"scenes": raw_scenes}

        other_attributes = {
            k: raw_data[k] for k in raw_data.column_names if k != "scenes"
        }

        # Normalize or inverse normalize the fetched scenes.
        scenes = raw_scenes if normalized else self.inverse_normalize_scenes(raw_scenes)

        if label is not None and "labels" in other_attributes:
            mask = other_attributes["labels"] == label
            scenes = scenes[mask]
            for key in other_attributes:
                other_attributes[key] = other_attributes[key][mask]

        other_attributes["scenes"] = scenes
        return other_attributes

    def set_data(self, data: dict[str, torch.Tensor], normalized: bool = False) -> None:
        """
        Replaces the dataset with new data.

        Note that this will disable subdataset sampling if it was enabled.

        Args:
            data (dict[str, torch.Tensor]): The data dictionary containing "scenes" and
                optionally additional attributes such as "labels".
            normalized (bool): Whether the scenes are normalized.
        """
        if "scenes" not in data:
            raise ValueError("The data dictionary must contain a 'scenes' key!")

        scenes = data["scenes"]

        # Normalize scenes if not already normalized.
        if not normalized:
            scenes = self.normalize_scenes(scenes)

        new_data = {"scenes": scenes.tolist()}
        for key, value in data.items():
            if key != "scenes":
                new_data[key] = (
                    value.tolist() if isinstance(value, torch.Tensor) else value
                )

        self.hf_dataset = Dataset.from_dict(new_data)
        self.hf_dataset.set_format("torch")

        if self.use_subdataset_sampling:
            self.use_subdataset_sampling = False
            console_logger.warning(
                "Subdataset sampling is disabled because the data was replaced."
            )

    def setup_subdataset_sampling(
        self, sampling_cfg: DictConfig, hf_dataset: Dataset
    ) -> None:
        """
        Sets up weighted subdataset sampling based on configuration.

        Args:
            sampling_cfg: Configuration for subdataset sampling.
            hf_dataset: HuggingFace dataset to sample from.
        """
        self.subdataset_probs = sampling_cfg.probabilities

        # Validate that all subdataset names have probabilities.
        if self.subdataset_probs is None:
            raise ValueError(
                "Subdataset sampling is enabled but no probabilities are specified."
            )
        for name in self.subdataset_names:
            if name not in self.subdataset_probs:
                raise ValueError(f"No probability specified for subdataset '{name}'")

        # Validate that probabilities approximately sum to 1.
        total_prob = sum(self.subdataset_probs.values())
        if not 0.98 <= total_prob <= 1.02:  # Allow for small floating point errors
            raise ValueError(
                f"Subdataset probabilities must sum to 1, got {total_prob}"
            )

        # Normalize probabilities to sum to 1.
        self.subdataset_probs = {
            name: prob / total_prob for name, prob in self.subdataset_probs.items()
        }

        # Create cumulative probabilities for efficient sampling
        self.subdataset_cum_probs = []
        cum_prob = 0.0
        for name in self.subdataset_names:
            cum_prob += self.subdataset_probs[name]
            self.subdataset_cum_probs.append(cum_prob)

        # Check if we should use infinite iterators.
        self.use_infinite_iterators = sampling_cfg.use_infinite_iterators
        if self.use_infinite_iterators:
            # Create infinite iterators for each subdataset.
            self.subdataset_iterators = []
            for start_idx, end_idx in self.subdataset_ranges:
                subdataset = hf_dataset.select(range(start_idx, end_idx))
                iterator = InfiniteShuffledStreamingDataset(
                    dataset=subdataset, buffer_size=sampling_cfg.buffer_size
                )
                self.subdataset_iterators.append(iterator)

            # Initialize iterators.
            self.subdataset_iterator_objects = [
                iter(iterator) for iterator in self.subdataset_iterators
            ]
            console_logger.info("Using infinite iterators for subdataset sampling.")

        console_logger.info(
            "Enabled weighted subdataset sampling with probabilities: "
            f"{self.subdataset_probs}"
        )

    def sample_subdataset_index(self) -> int:
        """
        Samples a subdataset index based on the configured probabilities.

        Returns:
            int: The index of the sampled subdataset.
        """
        if not self.use_subdataset_sampling:
            raise RuntimeError("Subdataset sampling is not enabled.")

        # Sample a random value between 0 and 1.
        r = random.random()

        # Find the subdataset whose cumulative probability range contains r.
        for i, cum_prob in enumerate(self.subdataset_cum_probs):
            if r <= cum_prob:
                return i

        # Something went wrong.
        raise RuntimeError("Failed to sample a subdataset index.")

    def sample_from_subdataset(self, subdataset_idx: int) -> int:
        """
        Samples an index from the specified subdataset.

        Args:
            subdataset_idx: Index of the subdataset to sample from.

        Returns:
            int: The global index of the sampled item.
        """
        # Use pre-computed indices for fast sampling.
        start_idx, end_idx = self.subdataset_ranges[subdataset_idx]
        return random.randint(start_idx, end_idx - 1)

    def get_subdataset_name_from_index(self, index: int) -> str:
        """
        Returns the subdataset name for the given dataset index.
        """
        if self.subdataset_ranges is None:
            raise ValueError("Subdataset ranges are not set!")

        for i, (start_idx, end_idx) in enumerate(self.subdataset_ranges):
            if index >= start_idx and index < end_idx:
                return self.subdataset_names[i]
        raise ValueError(f"Index {index} is out of range for any subdataset.")

    def _validate_subdataset_config(self) -> None:
        """
        Validates the subdataset configuration, ensuring that necessary metadata
        is available when using subdataset features.
        """
        # Check if static subdataset prompts are enabled but metadata is missing.
        if self.cfg.static_subdataset_prompts.use and (
            self.subdataset_ranges is None or self.subdataset_names is None
        ):
            raise ValueError(
                "Require subdataset ranges and names to be set when using static "
                "subdataset prompts!"
            )

        # Check if subdataset sampling is enabled but metadata is missing.
        if self.cfg.subdataset_sampling.use and (
            self.subdataset_ranges is None or self.subdataset_names is None
        ):
            raise ValueError(
                "Require subdataset ranges and names to be set when using "
                "subdataset sampling!"
            )

        # Check if static prompts are provided for all subdatasets.
        if self.cfg.static_subdataset_prompts.use and set(
            self.cfg.static_subdataset_prompts.name_to_prompt.keys()
        ) != set(self.subdataset_names):
            raise ValueError(
                "Require static subdataset prompts to be set for all subdatasets!\n"
                f"Subdataset names: {self.subdataset_names}\n"
                f"Prompts: {self.cfg.static_subdataset_prompts.name_to_prompt.keys()}"
            )

    def _validate_dataset_structure(
        self, hf_dataset: Dataset, metadata: dict[str, Any]
    ) -> None:
        """
        Validates the dataset structure against configuration.

        Args:
            hf_dataset (Dataset): The loaded HuggingFace dataset.
            metadata (dict[str, Any]): The dataset metadata.
        """
        scene_shape = hf_dataset[0]["scenes"].shape
        if not scene_shape[0] == self.cfg.max_num_objects_per_scene:
            raise ValueError(
                f"The number of dataset scene objects ({scene_shape[0]}) does not "
                f"match the config ({self.cfg.max_num_objects_per_scene})."
            )
        if (
            self.cfg.model_path_vec_len is not None
            and self.cfg.model_path_vec_len != metadata["model_path_vec_len"]
        ):
            raise ValueError(
                f"The dataset model path vector length ({metadata['model_path_vec_len']}) "
                f"does not match the config ({self.cfg.model_path_vec_len})."
            )
