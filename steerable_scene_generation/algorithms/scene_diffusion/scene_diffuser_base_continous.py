import logging

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import torch

from diffusers import DDIMScheduler
from tqdm import tqdm

from steerable_scene_generation.datasets.scene.scene import SceneDataset
from steerable_scene_generation.utils.caching import conditional_cache

from .scene_diffuser_base import SceneDiffuserBase

logger = logging.getLogger(__name__)


class SceneDiffuserBaseContinous(SceneDiffuserBase, ABC):
    """
    Abstract base class for continous diffusion on scene vectors. This builds on top of
    `SceneDiffuserBase` to add continous diffusion specific functionality.
    """

    def __init__(self, cfg, dataset: SceneDataset):
        """
        cfg is a DictConfig object defined by
        `configurations/algorithm/scene_diffuser_base_continous.yaml`.
        """
        super().__init__(cfg, dataset=dataset)

    @abstractmethod
    def predict_noise(
        self,
        noisy_scenes: torch.Tensor,
        timesteps: Union[torch.IntTensor, int],
        cond_dict: Dict[str, Any] = None,
        use_ema: bool = False,
    ) -> torch.Tensor:
        """Predict the noise for a batch of noisy scenes.

        Args:
            noisy_scenes (torch.Tensor): Input of shape (B, N, V) where N are the
                number of objects and V is the object feature vector length.
            timesteps (Union[torch.IntTensor, int]): The diffusion step to condition
                the denoising on.
            cond_dict: Dict[str, Any]: The dict containing the conditioning information.
            use_ema (bool): Whether to use the EMA model.

        Returns:
            torch.Tensor: Output of same shape as the input.
        """
        raise NotImplementedError

    def sample_scenes_without_guidance(
        self,
        num_samples: int,
        use_ema: bool = False,
    ) -> torch.Tensor:
        """
        Sample scenes from the model without guidance. The scenes are inverse
        normalized.
        """
        if isinstance(self.noise_scheduler, DDIMScheduler):
            self.noise_scheduler.set_timesteps(
                self.cfg.noise_schedule.ddim.num_inference_timesteps, device=self.device
            )

        # Sample random noise.
        num_objects_per_scene = (
            self.cfg.max_num_objects_per_scene
            + self.cfg.num_additional_tokens_for_sampling
        )
        xt = self.sample_continuous_noise_prior(
            (
                num_samples,
                num_objects_per_scene,
                self.scene_vec_desc.get_object_vec_len(),
            )
        )  # Shape (B, N, V)

        for t in tqdm(
            self.noise_scheduler.timesteps, desc="Sampling scenes", leave=False
        ):
            with torch.no_grad():
                residual = self.predict_noise(
                    xt, t, cond_dict=None, use_ema=use_ema
                )  # Shape (B, N, V)

            # Compute the updated sample.
            if isinstance(self.noise_scheduler, DDIMScheduler):
                scheduler_out = self.noise_scheduler.step(
                    residual, t, xt, eta=self.cfg.noise_schedule.ddim.eta
                )
            else:
                scheduler_out = self.noise_scheduler.step(residual, t, xt)

            # Update the sample.
            xt = scheduler_out.prev_sample  # Shape (B, N, V)

            if self.cfg.visualization.visualize_intermediate_scenes:
                self.visualize_intermediate_scene(t, xt)

        # Apply inverse normalization.
        xt = self.dataset.inverse_normalize_scenes(xt)  # Shape (B, N, V)

        return xt

    def sample_scenes_with_classifier_free_guidance(
        self, num_samples: int, cond_dict: Dict[str, Any] = None, use_ema: bool = False
    ) -> torch.Tensor:
        """
        Sample scenes from the model with classifier-free guidance.

        Args:
            num_samples (int): The number of samples to generate.
            cond_dict: Dict[str, Any]: The dict containing the conditioning information.
            use_ema (bool): Whether to use the EMA model.

        Returns:
            torch.Tensor: The generated samples of shape (B, N, V). The samples are
                unormalized.
        """
        if cond_dict is not None:
            # Add the mask labels to the cond_dict.
            cond_dict = self.dataset.add_classifier_free_guidance_uncond_data(
                cond_dict.copy()
            )

        if isinstance(self.noise_scheduler, DDIMScheduler):
            self.noise_scheduler.set_timesteps(
                self.cfg.noise_schedule.ddim.num_inference_timesteps, device=self.device
            )

        # Sample random noise.
        num_objects_per_scene = (
            self.cfg.max_num_objects_per_scene
            + self.cfg.num_additional_tokens_for_sampling
        )
        xt = self.sample_continuous_noise_prior(
            (
                num_samples,
                num_objects_per_scene,
                self.scene_vec_desc.get_object_vec_len(),
            )
        )  # Shape (B, N, V)

        for t in tqdm(
            self.noise_scheduler.timesteps,
            desc="Sampling scenes with classifier-free guidance",
            leave=False,
        ):
            with torch.no_grad():
                prediction = self.predict_noise(
                    xt.repeat(2, 1, 1), t, cond_dict=cond_dict, use_ema=use_ema
                )  # Shape (B*2, N, V)
                cond_pred = prediction[:num_samples]  # Shape (B, N, V)
                uncond_pred = prediction[num_samples:]  # Shape (B, N, V)

                # Residual has shape (B, N, V).
                weight = self.cfg.classifier_free_guidance.weight
                residual = (1 + weight) * cond_pred - weight * uncond_pred

            # Compute the updated sample.
            if isinstance(self.noise_scheduler, DDIMScheduler):
                scheduler_out = self.noise_scheduler.step(
                    residual, t, xt, eta=self.cfg.noise_schedule.ddim.eta
                )
            else:
                scheduler_out = self.noise_scheduler.step(residual, t, xt)

            # Update the sample.
            xt = scheduler_out.prev_sample  # Shape (B, N, V)

        # Apply inverse normalization.
        xt = self.dataset.inverse_normalize_scenes(xt)  # Shape (B, N, V)

        return xt

    @torch.no_grad
    def sample_scenes_continous_or_discrete_only(
        self,
        num_samples: int,
        data_batch: Dict[str, torch.Tensor],
        use_ema: bool = False,
    ) -> torch.Tensor:
        """
        Sample scenes from the model where the continous or discrete part is kept from
        the scenes in `data_batch`.

        Args:
            num_samples (int): The number of samples to generate.
            data_batch (Dict[str, torch.Tensor]): The data batch that contains scenes of
                shape (M, N, V). Note that M must be greater or equal than
                `num_samples`.
            use_ema (bool): Whether to use the EMA model.

        Returns:
            torch.Tensor: The generated samples of shape (B, N, V). The samples are
                unormalized.
        """
        assert (
            self.cfg.continuous_discrete_only.continuous_only
            or self.cfg.continuous_discrete_only.discrete_only
        )
        scene_data_batch = data_batch["scenes"]
        assert len(scene_data_batch) >= num_samples
        scene_data_batch = scene_data_batch.to(self.device)

        if self.cfg.classifier_free_guidance.use:
            # Add the mask labels to the cond_dict.
            data_batch = self.dataset.add_classifier_free_guidance_uncond_data(
                data_batch.copy()
            )

        if isinstance(self.noise_scheduler, DDIMScheduler):
            self.noise_scheduler.set_timesteps(
                self.cfg.noise_schedule.ddim.num_inference_timesteps, device=self.device
            )

        # Sample random noise for the continous or discrete part while taking the other
        # part from the data batch.
        if self.cfg.num_additional_tokens_for_sampling > 0:
            raise NotImplementedError(
                "Sampling from the continous or discrete part only is not implemented "
                "when there are additional tokens for sampling."
            )
        xt = self.sample_continuous_noise_prior(
            (
                num_samples,
                self.cfg.max_num_objects_per_scene,
                self.scene_vec_desc.get_object_vec_len(),
            )
        )  # Shape (B, N, V)
        if self.cfg.continuous_discrete_only.continuous_only:
            mask = torch.concatenate(
                [
                    torch.ones(
                        self.scene_vec_desc.translation_vec_len
                        + len(self.scene_vec_desc.rotation_parametrization)
                    ),
                    torch.zeros(self.scene_vec_desc.model_path_vec_len),
                ]
            ).to(self.device)
        elif self.cfg.continuous_discrete_only.discrete_only:
            mask = torch.concatenate(
                [
                    torch.zeros(
                        self.scene_vec_desc.translation_vec_len
                        + len(self.scene_vec_desc.rotation_parametrization)
                    ),
                    torch.ones(self.scene_vec_desc.model_path_vec_len),
                ]
            ).to(self.device)
        mask_expanded = (
            mask.unsqueeze(0).unsqueeze(0).expand(xt.shape)
        )  # Shape (B, N, V)
        xt = xt * mask_expanded + scene_data_batch[:num_samples] * (1 - mask_expanded)

        for t in tqdm(
            self.noise_scheduler.timesteps,
            desc="Sampling scenes (continuous or discrete part only)",
            leave=False,
        ):
            if self.cfg.classifier_free_guidance.use:
                prediction = self.predict_noise(
                    xt.repeat(2, 1, 1), t, cond_dict=data_batch, use_ema=use_ema
                )  # Shape (B*2, N, V)
                cond_pred = prediction[:num_samples]  # Shape (B, N, V)
                uncond_pred = prediction[num_samples:]  # Shape (B, N, V)

                # Residual has shape (B, N, V).
                weight = self.cfg.classifier_free_guidance.weight
                residual = (1 + weight) * cond_pred - weight * uncond_pred
            else:
                residual = self.predict_noise(xt, t, use_ema=use_ema)  # Shape (B, N, V)

            # Compute the updated sample.
            if isinstance(self.noise_scheduler, DDIMScheduler):
                scheduler_out = self.noise_scheduler.step(
                    residual, t, xt, eta=self.cfg.noise_schedule.ddim.eta
                )
            else:
                scheduler_out = self.noise_scheduler.step(residual, t, xt)

            # Update the discrete/ continuous part of the sample.
            xt = scheduler_out.prev_sample * mask_expanded + xt * (1 - mask_expanded)

        # Apply inverse normalization.
        xt = self.dataset.inverse_normalize_scenes(xt)  # Shape (B, N, V)

        return xt

    @conditional_cache(argument_name="is_test")  # Only cache during test time.
    def sample_scenes(
        self,
        num_samples: int,
        is_test: bool = False,
        batch_size: Optional[int] = None,
        use_ema: bool = False,
        data_batch: Dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if batch_size is None:
            batch_size = num_samples

        # Determine the batches in which to sample the scenes.
        num_batches = num_samples // batch_size
        remainder = num_samples % batch_size
        batch_sizes = [batch_size] * num_batches
        if remainder > 0:
            batch_sizes.append(remainder)

        if (
            data_batch is not None
            and self.cfg.classifier_free_guidance.use
            and len(data_batch["text_cond"]["input_ids"]) != num_samples
        ):
            # Ensure the data batch size matches the number of samples.
            if len(data_batch["text_cond"]["input_ids"]) > num_samples:
                # Take first num_samples samples.
                data_batch = {k: v[:num_samples] for k, v in data_batch.items()}
            else:
                logger.warning(
                    f"Data batch size {len(data_batch['text_cond']['input_ids'])} is "
                    f"smaller than the number of samples {num_samples}. Sampling "
                    f"the first {len(data_batch['text_cond']['input_ids'])} samples."
                )
                data_batch = self.dataset.sample_data_dict(
                    data=data_batch, num_items=num_samples
                )

        # Optionally replace the data labels with the specified labels during testing.
        if (
            is_test
            and not self.cfg.classifier_free_guidance.sampling.use_data_labels
            and data_batch is not None
        ):
            txt_labels = self.cfg.classifier_free_guidance.sampling.labels
            data_batch = self.dataset.replace_cond_data(
                data=data_batch, txt_labels=txt_labels
            )

        # Sample scenes.
        sampled_scene_batches = []
        for num in tqdm(batch_sizes, desc="Sampling scene batches", leave=False):
            if (
                self.cfg.continuous_discrete_only.continuous_only
                or self.cfg.continuous_discrete_only.discrete_only
            ):
                scenes = self.sample_scenes_continous_or_discrete_only(
                    num, data_batch=data_batch, use_ema=use_ema
                )
            elif self.cfg.classifier_free_guidance.use:
                scenes = self.sample_scenes_with_classifier_free_guidance(
                    num, cond_dict=data_batch, use_ema=use_ema
                )
            else:
                scenes = self.sample_scenes_without_guidance(num, use_ema=use_ema)
            sampled_scene_batches.append(scenes)
        sampled_scenes = torch.cat(sampled_scene_batches, dim=0)

        if is_test:
            # Compute sampled scene metrics.
            self.log_sampled_scene_metrics(
                sampled_scenes, name="sampled_scenes/before_processing"
            )

        # Apply post-processing.
        sampled_scenes = self.apply_postprocessing(sampled_scenes).to(self.device)

        if is_test:
            # Compute sampled scene metrics.
            self.log_sampled_scene_metrics(
                sampled_scenes, name="sampled_scenes/after_processing"
            )

        return sampled_scenes

    def inpaint_scenes(
        self, data_batch: Dict[str, torch.Tensor], use_ema: bool = False
    ) -> torch.Tensor:
        # Extract scenes and masks from the data batch.
        scenes = data_batch["scenes"]  # Shape (B, N, V)
        inpainting_masks = data_batch["inpainting_masks"]  # Shape (B, N, V)

        if not scenes.shape == inpainting_masks.shape:
            raise ValueError(
                "Scenes and inpainting masks must have the same shape. "
                f"Got {scenes.shape} and {inpainting_masks.shape}."
            )

        # Set timesteps for inference.
        if isinstance(self.noise_scheduler, DDIMScheduler):
            self.noise_scheduler.set_timesteps(
                self.cfg.noise_schedule.ddim.num_inference_timesteps, device=self.device
            )

        # Initialize with random noise for masked regions.
        xt = self.sample_continuous_noise_prior(scenes.shape)  # Shape (B, N, V)
        xt = torch.where(inpainting_masks, xt, scenes)  # Apply mask

        if self.cfg.classifier_free_guidance.use:
            # Add the mask labels to the data_batch.
            data_batch = self.dataset.add_classifier_free_guidance_uncond_data(
                data_batch.copy()
            )

        num_samples = scenes.shape[0]
        for t in tqdm(
            self.noise_scheduler.timesteps, desc="Inpainting scenes", leave=False
        ):
            with torch.no_grad():
                if self.cfg.classifier_free_guidance.use:
                    # Double the batch for classifier-free guidance.
                    noise_pred = self.predict_noise(
                        noisy_scenes=xt.repeat(2, 1, 1),
                        timesteps=t,
                        cond_dict=data_batch,
                        use_ema=use_ema,
                    )  # Shape (B*2, N, V)

                    noise_pred_cond = noise_pred[:num_samples]  # Shape (B, N, V)
                    noise_pred_uncond = noise_pred[num_samples:]  # Shape (B, N, V)

                    # Apply classifier-free guidance.
                    weight = self.cfg.classifier_free_guidance.weight
                    noise_pred = (
                        1 + weight
                    ) * noise_pred_cond - weight * noise_pred_uncond
                else:
                    noise_pred = self.predict_noise(
                        noisy_scenes=xt,
                        timesteps=t,
                        cond_dict=data_batch,
                        use_ema=use_ema,
                    )  # Shape (B, N, V)

            # Update the sample for masked regions.
            if isinstance(self.noise_scheduler, DDIMScheduler):
                scheduler_out = self.noise_scheduler.step(
                    noise_pred, t, xt, eta=self.cfg.noise_schedule.ddim.eta
                )
            else:
                scheduler_out = self.noise_scheduler.step(noise_pred, t, xt)

            xt_next = scheduler_out.prev_sample  # Shape (B, N, V)

            # Only update masked regions, keep unmasked regions fixed.
            xt = torch.where(inpainting_masks, xt_next, scenes)

        # Apply inverse normalization.
        inpainted_scenes = self.dataset.inverse_normalize_scenes(xt)  # Shape (B, N, V)

        # Compute inpainted scene metrics.
        self.log_sampled_scene_metrics(
            inpainted_scenes, name="inpainted_scenes/before_processing"
        )

        # Apply post-processing.
        inpainted_scenes = self.apply_postprocessing(inpainted_scenes).to(self.device)

        # Compute inpainted scene metrics after processing.
        self.log_sampled_scene_metrics(
            inpainted_scenes, name="inpainted_scenes/after_processing"
        )

        return inpainted_scenes
