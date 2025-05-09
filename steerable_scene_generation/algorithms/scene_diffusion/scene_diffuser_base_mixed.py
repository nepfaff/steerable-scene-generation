import logging

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import torch
import torch.nn.functional as F

from diffusers import DDIMScheduler
from tqdm import tqdm

from steerable_scene_generation.algorithms.common.discrete_diffusion import (
    MaskAndReplaceDiffusion,
)
from steerable_scene_generation.datasets.scene.scene import SceneDataset
from steerable_scene_generation.utils.caching import conditional_cache

from .scene_diffuser_base import SceneDiffuserBase

logger = logging.getLogger(__name__)


class SceneDiffuserBaseMixed(SceneDiffuserBase, ABC):
    """
    Abstract base class for mixed continous and discrete diffusion on scene vectors.
    This builds on top of `SceneDiffuserBase` to add mixed diffusion specific
    functionality.
    """

    def __init__(self, cfg, dataset: SceneDataset):
        """
        cfg is a DictConfig object defined by
        `configurations/algorithm/scene_diffuser_base_mixed.yaml`.
        """
        super().__init__(cfg, dataset=dataset)

    def _build_model(self):
        """Create the discrete diffusion method class."""
        super()._build_model()

        # Includes the [empty] and [mask] token.
        self.num_diffusion_classes = self.cfg.model_path_vec_len + 1
        self.mask_token_idx = self.cfg.model_path_vec_len
        self.discrete_diffusion = MaskAndReplaceDiffusion(
            num_classes=self.num_diffusion_classes,
            num_timesteps=self.cfg.noise_schedule.num_train_timesteps,
        )

    @abstractmethod
    def denoise(
        self,
        x_continous: torch.Tensor,
        x_discrete: torch.Tensor,
        timesteps: Union[torch.IntTensor, int],
        cond_dict: Dict[str, Any] = None,
        use_ema: bool = False,
    ) -> Union[torch.Tensor, torch.Tensor]:
        """For a batch of noisy scenes, predict the noise for the continous part and
        x0 for the discrete part.

        Args:
            x_continous (torch.Tensor): Continuous scenes of shape (B, N, Vc) where
                N are the number of objects and Vc is the continous object feature
                vector length.
            x_discrete (torch.Tensor): The discrete input of shape (B, N) where
                values are non-zero integers that represent discrete classes.
            timesteps (Union[torch.IntTensor, int]): The diffusion step to condition
                the denoising on.
            cond_dict: Dict[str, Any]: The dict containing the conditioning information.
            use_ema (bool): Whether to use the EMA model.

        Returns:
            Union[torch.Tensor, torch.Tensor]: A tuple of
                - continous_output: Noise prediction for `x_continous` of of shape
                    (B, N, Vc).
                - discrete_output: x0 prediction for `x_discrete` of shape (B,
                    num_discrete_diffusion_classes-1, N). This doesn't include the
                    class for the [mask] token.
        """
        raise NotImplementedError

    @conditional_cache(argument_name="is_test")  # Only cache during test time.
    def sample_scenes(
        self,
        num_samples: int,
        is_test: bool = False,
        batch_size: Optional[int] = None,
        use_ema: bool = False,
        data_batch: Dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if batch_size is not None:
            raise NotImplementedError(
                "Sampling in batches is not currently implemented."
            )
        if (
            isinstance(self.noise_scheduler, DDIMScheduler)
            and self.cfg.noise_schedule.ddim.num_inference_timesteps
            != self.cfg.noise_schedule.num_train_timesteps
        ):
            raise NotImplementedError(
                "Mixed sampling doesn't yet support continous schedule with fewer "
                "inference than train steps."
            )

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

        if is_test and self.cfg.classifier_free_guidance.use and data_batch is not None:
            # Add the mask labels to the data_batch.
            data_batch = self.dataset.add_classifier_free_guidance_uncond_data(
                data_batch.copy()
            )

        # Sample continuous random noise.
        num_objects_per_scene = (
            self.cfg.max_num_objects_per_scene
            + self.cfg.num_additional_tokens_for_sampling
        )
        c_xt = self.sample_continuous_noise_prior(
            (
                num_samples,
                num_objects_per_scene,
                self.scene_vec_desc.get_object_vec_len()
                - self.scene_vec_desc.model_path_vec_len,
            )
        ).to(
            self.device
        )  # Shape (B, N, Vc)

        # Sample discrete noise.
        if self.cfg.discrete_sampling.use_all_mask_as_noise:
            d_xt = (
                torch.full(
                    (num_samples, self.cfg.max_num_objects_per_scene),
                    self.mask_token_idx,
                )
                .long()
                .to(self.device)
            )  # Shape (B, N)
        else:
            d_xt = torch.randint(
                low=0,
                high=self.mask_token_idx,
                size=(num_samples, self.cfg.max_num_objects_per_scene),
            ).to(
                self.device
            )  # Shape (B, N)
        d_log_xt = self.discrete_diffusion.index_to_log_onehot(
            d_xt, num_classes=self.num_diffusion_classes
        )  # Shape (B, Vd+1, N), includes [mask] token

        for t in tqdm(
            self.noise_scheduler.timesteps, desc="Sampling scenes", leave=False
        ):
            with torch.no_grad():
                if is_test and self.cfg.classifier_free_guidance.use:
                    c_pred, d_pred = self.denoise(
                        x_continous=c_xt.repeat(2, 1, 1),
                        x_discrete=d_xt.repeat(2, 1),
                        timesteps=t,
                        cond_dict=data_batch,
                        use_ema=use_ema,
                    )  # First has shape (B*2, N, Vc) and second has shape (B*2, Vd, N)

                    c_cond_pred = c_pred[:num_samples]  # Shape (B, N, Vc)
                    c_uncond_pred = c_pred[num_samples:]  # Shape (B, N, Vc)
                    d_cond_pred = d_pred[:num_samples]  # Shape (B, Vd, N)
                    d_uncond_pred = d_pred[num_samples:]  # Shape (B, Vd, N)

                    weight = self.cfg.classifier_free_guidance.weight
                    c_pred = (1 + weight) * c_cond_pred - weight * c_uncond_pred
                    d_pred = (1 + weight) * d_cond_pred - weight * d_uncond_pred
                else:
                    c_pred, d_pred = self.denoise(
                        x_continous=c_xt,
                        x_discrete=d_xt,
                        timesteps=t,
                        cond_dict=data_batch,
                        use_ema=use_ema,
                    )  # First has shape (B, N, Vc) and second has shape (B, Vd, N)

            # Update the continous sample.
            if isinstance(self.noise_scheduler, DDIMScheduler):
                scheduler_out = self.noise_scheduler.step(
                    c_pred, t, c_xt, eta=self.cfg.noise_schedule.ddim.eta
                )
            else:
                scheduler_out = self.noise_scheduler.step(c_pred, t, c_xt)
            c_xt = scheduler_out.prev_sample  # Shape (B, N, Vc)

            # Update the discrete sample.
            d_log_x_recon = self.discrete_diffusion.log_pred_from_denoise_out(
                d_pred
            )  # Shape (B, Vd+1, N)
            if t == 0:
                # Don't want mask tokens in the final sample.
                d_log_EV_qxt_x0 = d_log_x_recon  # Shape (B, Vd+1, N)
            else:
                d_log_EV_qxt_x0 = self.discrete_diffusion.q_posterior(
                    log_x0_recon=d_log_x_recon,
                    log_x_t=d_log_xt,
                    t=t.unsqueeze(0).expand(num_samples).to(self.device),
                )  # Shape (B, Vd+1, N)
            d_log_xt = self.discrete_diffusion.log_sample_categorical(
                d_log_EV_qxt_x0
            )  # Shape (B, Vd+1, N)
            d_xt = self.discrete_diffusion.log_onehot_to_index(d_log_xt)  # Shape (B, N)

            if self.cfg.visualization.visualize_intermediate_scenes:
                # Ensure that the mask tokens are not in the sample for visualization.
                d_log_xt_tmp = self.discrete_diffusion.log_sample_categorical(
                    d_log_x_recon
                )  # Shape (B, Vd+1, N)
                d_xt_tmp = self.discrete_diffusion.log_onehot_to_index(
                    d_log_xt_tmp
                )  # Shape (B, N)
                d_xt_tmp = d_xt_tmp.detach().cpu().clone()  # Shape (B, N)
                d_xt_onehot_tmp = F.one_hot(
                    d_xt_tmp, self.num_diffusion_classes - 1  # Exclude [mask] token.
                )  # Shape (B, N, Vd)
                xt_tmp = torch.concat(
                    [c_xt.detach().cpu().clone(), d_xt_onehot_tmp], dim=-1
                )  # Shape (B, N, Vc + Vd)
                self.visualize_intermediate_scene(t, xt_tmp)

        # Combine continous and discrete parts.
        d_xt_onehot = F.one_hot(
            d_xt, self.num_diffusion_classes - 1  # Exclude [mask] token.
        )  # Shape (B, N, Vd)
        xt = torch.concat([c_xt, d_xt_onehot], dim=-1)  # Shape (B, N, Vc + Vd)

        # Apply inverse normalization.
        sampled_scenes = self.dataset.inverse_normalize_scenes(
            xt
        )  # Shape (B, N, Vc + Vd)

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
        if (
            isinstance(self.noise_scheduler, DDIMScheduler)
            and self.cfg.noise_schedule.ddim.num_inference_timesteps
            != self.cfg.noise_schedule.num_train_timesteps
        ):
            raise NotImplementedError(
                "Mixed inpainting doesn't yet support continuous schedule with fewer "
                "inference than train steps."
            )

        # Extract scenes and masks from the data batch.
        scenes = data_batch["scenes"]  # Shape (B, N, V)
        inpainting_masks = data_batch["inpainting_masks"]  # Shape (B, N, V)

        if not scenes.shape == inpainting_masks.shape:
            raise ValueError(
                "Scenes and inpainting masks must have the same shape. "
                f"Got {scenes.shape} and {inpainting_masks.shape}."
            )

        # Split scenes into continuous and discrete parts.
        continuous_part_len = (
            self.scene_vec_desc.get_object_vec_len()
            - self.scene_vec_desc.model_path_vec_len
        )
        c_scenes = scenes[..., :continuous_part_len]  # Shape (B, N, Vc)
        d_scenes_onehot = scenes[..., continuous_part_len:]  # Shape (B, N, Vd)

        # Convert discrete part from one-hot to indices.
        d_scenes = torch.argmax(d_scenes_onehot, dim=-1)  # Shape (B, N)

        # Split masks into continuous and discrete parts.
        c_masks = inpainting_masks[..., :continuous_part_len]  # Shape (B, N, Vc)
        d_masks = inpainting_masks[..., continuous_part_len:]  # Shape (B, N, Vd)

        # Create a mask for the entire discrete part if any discrete element needs
        # inpainting.
        d_masks_per_object = d_masks.any(dim=-1)  # Shape (B, N)

        # Initialize continuous part with random noise for masked regions.
        c_xt = torch.randn_like(c_scenes)  # Shape (B, N, Vc)
        c_xt = torch.where(c_masks, c_xt, c_scenes)  # Apply mask

        # Initialize discrete part with random noise for masked regions.
        if self.cfg.discrete_sampling.use_all_mask_as_noise:
            d_noise = torch.full_like(d_scenes, self.mask_token_idx)
        else:
            d_noise = torch.randint(
                low=0,
                high=self.mask_token_idx,
                size=d_scenes.shape,
                device=d_scenes.device,
            )
        d_xt = torch.where(d_masks_per_object, d_noise, d_scenes)  # Apply mask

        # Convert to log one-hot.
        d_log_xt = self.discrete_diffusion.index_to_log_onehot(
            d_xt, num_classes=self.num_diffusion_classes
        )  # Shape (B, Vd+1, N), includes [mask] token

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
                    c_pred, d_pred = self.denoise(
                        x_continous=c_xt.repeat(2, 1, 1),
                        x_discrete=d_xt.repeat(2, 1),
                        timesteps=t,
                        cond_dict=data_batch,
                        use_ema=use_ema,
                    )  # First has shape (B*2, N, Vc) and second has shape (B*2, Vd, N)

                    c_cond_pred = c_pred[:num_samples]  # Shape (B, N, Vc)
                    c_uncond_pred = c_pred[num_samples:]  # Shape (B, N, Vc)
                    d_cond_pred = d_pred[:num_samples]  # Shape (B, Vd, N)
                    d_uncond_pred = d_pred[num_samples:]  # Shape (B, Vd, N)

                    weight = self.cfg.classifier_free_guidance.weight
                    c_pred = (1 + weight) * c_cond_pred - weight * c_uncond_pred
                    d_pred = (1 + weight) * d_cond_pred - weight * d_uncond_pred
                else:
                    c_pred, d_pred = self.denoise(
                        x_continous=c_xt,
                        x_discrete=d_xt,
                        timesteps=t,
                        cond_dict=data_batch,
                        use_ema=use_ema,
                    )  # First has shape (B, N, Vc) and second has shape (B, Vd, N)

            # Update the continuous sample for masked regions.
            if isinstance(self.noise_scheduler, DDIMScheduler):
                scheduler_out = self.noise_scheduler.step(
                    c_pred, t, c_xt, eta=self.cfg.noise_schedule.ddim.eta
                )
            else:
                scheduler_out = self.noise_scheduler.step(c_pred, t, c_xt)
            c_xt_next = scheduler_out.prev_sample  # Shape (B, N, Vc)

            # Only update masked regions, keep unmasked regions fixed.
            c_xt = torch.where(c_masks, c_xt_next, c_scenes)

            # Update the discrete sample for masked regions.
            d_log_x_recon = self.discrete_diffusion.log_pred_from_denoise_out(
                d_pred
            )  # Shape (B, Vd+1, N)

            if t == 0:
                d_log_EV_qxt_x0 = d_log_x_recon  # Shape (B, Vd+1, N)
            else:
                d_log_EV_qxt_x0 = self.discrete_diffusion.q_posterior(
                    log_x0_recon=d_log_x_recon,
                    log_x_t=d_log_xt,
                    t=t.unsqueeze(0).expand(num_samples).to(self.device),
                )  # Shape (B, Vd+1, N)

            d_log_xt_next = self.discrete_diffusion.log_sample_categorical(
                d_log_EV_qxt_x0
            )  # Shape (B, Vd+1, N)
            d_xt_next = self.discrete_diffusion.log_onehot_to_index(
                d_log_xt_next
            )  # Shape (B, N)

            # Only update masked regions, keep unmasked regions fixed.
            d_xt = torch.where(d_masks_per_object, d_xt_next, d_scenes)

            # Update log one-hot representation for next iteration.
            d_log_xt = self.discrete_diffusion.index_to_log_onehot(
                d_xt, num_classes=self.num_diffusion_classes
            )

        # Combine continuous and discrete parts.
        d_xt_onehot = F.one_hot(
            d_xt, self.num_diffusion_classes - 1  # Exclude [mask] token
        )  # Shape (B, N, Vd)
        xt = torch.concat([c_xt, d_xt_onehot], dim=-1)  # Shape (B, N, Vc + Vd)

        # Apply inverse normalization.
        inpainted_scenes = self.dataset.inverse_normalize_scenes(
            xt
        )  # Shape (B, N, Vc + Vd)

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
