from typing import Dict

import torch
import torch.nn.functional as F

from steerable_scene_generation.datasets.scene.scene import SceneDataset

from .scene_diffuser_base_mixed import SceneDiffuserBaseMixed


class SceneDiffuserTrainerMixedDiffusion(SceneDiffuserBaseMixed):
    """
    Class that provides the mixed DDPM and D3PM training logic as proposed in
    https://arxiv.org/abs/2405.21066.
    """

    def __init__(self, cfg, dataset: SceneDataset):
        """
        cfg is a DictConfig object defined by
        `configurations/algorithm/scene_diffuser_base_mixed.yaml`.
        """
        super().__init__(cfg, dataset=dataset)

        obj_vec_len = self.scene_vec_desc.get_object_vec_len()
        discrete_vec_len = self.scene_vec_desc.model_path_vec_len
        self.continous_vec_len = obj_vec_len - discrete_vec_len

    def continous_loss_function(
        self, predicted_noise: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the continous DDPM loss.

        Args:
            predicted_noise (torch.Tensor): Predicted noise of shape (B, N, V).
            noise (torch.Tensor): GT noise of shape (B, N, V).

        Return:
            torch.Tensor: The loss item tensor.
        """
        if self.cfg.loss.use_separate_loss_per_object_attribute:
            batch_size = predicted_noise.shape[0]

            # Ensure that each object attribute is weighted equally, regardless of its
            # number of parameters.
            translation_loss = F.mse_loss(
                self.scene_vec_desc.get_translation_vec(predicted_noise),
                self.scene_vec_desc.get_translation_vec(noise),
            )
            rotation_loss = F.mse_loss(
                self.scene_vec_desc.get_rotation_vec(predicted_noise),
                self.scene_vec_desc.get_rotation_vec(noise),
            )
            loss = (
                self.cfg.loss.object_translation_attribute_weight * translation_loss
                + self.cfg.loss.object_rotation_attribute_weight * rotation_loss
            )
            # Normalize the loss for the scaling not to affect the learning rate.
            loss /= (
                self.cfg.loss.object_translation_attribute_weight
                + self.cfg.loss.object_rotation_attribute_weight
            )
            self.log_dict(
                {
                    "training/translation_loss": translation_loss,
                    "training/rotation_loss": rotation_loss,
                },
                batch_size=batch_size,
            )
        else:
            loss = F.mse_loss(predicted_noise, noise)

        return loss

    def discrete_loss_function(
        self,
        log_x0: torch.Tensor,
        log_xt: torch.Tensor,
        log_x0_recon: torch.Tensor,
        log_pred_prob: torch.Tensor,
        timesteps: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Computes the discrete D3PM loss.

        Args:
            log_x0 (torch.Tensor): The log one-hot vector of the initial state of shape
                (B, C, N).
            log_xt (torch.Tensor): The log one-hot vector of the noised state of shape
                (B, C, N).
            log_x0_recon (torch.Tensor): The reconstructed log one-hot vector of the
                initial state (model output) of shape (B, C, N).
            log_pred_prob (torch.Tensor): The log probabilities for the predicted
                classes at timestep `t-1` of shape (B, C, N).
            timesteps (torch.LongTensor): The timesteps of shape (B,).

        Return:
            torch.Tensor: The loss item tensor.
        """
        batch_size = log_x0.shape[0]

        # Compute KL divergence loss.
        kl_loss = self.discrete_diffusion.compute_kl_loss(
            log_x_0=log_x0, log_x_t=log_xt, t=timesteps, log_pred_prob=log_pred_prob
        ).mean()  # Shape (1,)

        # Compute auxiliary loss.
        if self.cfg.auxiliary_loss.weight > 0.0:
            # aux_loss of shape (B,).
            aux_loss = self.discrete_diffusion.compute_aux_loss(
                log_x_0=log_x0, log_x0_recon=log_x0_recon, t=timesteps
            ).mean(dim=1)
            if self.cfg.auxiliary_loss.adaptive:
                adaptive_aux_loss_weight = (
                    1.0 - timesteps / self.discrete_diffusion.num_timesteps
                ) + 1.0
            else:
                adaptive_aux_loss_weight = 1.0

            aux_loss = (
                adaptive_aux_loss_weight * self.cfg.auxiliary_loss.weight * aux_loss
            ).mean()
        else:
            aux_loss = torch.tensor(0.0)

        self.log_dict(
            {"training/kl_loss": kl_loss, "training/aux_loss": aux_loss},
            batch_size=batch_size,
        )
        loss = kl_loss + aux_loss
        return loss

    def loss_function(
        self,
        c_predicted_noise: torch.Tensor,
        c_noise: torch.Tensor,
        d_log_x0: torch.Tensor,
        d_log_xt: torch.Tensor,
        d_log_x0_recon: torch.Tensor,
        d_log_pred_prob: torch.Tensor,
        timesteps: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Compute the loss function for both continous `c_` and discrete `d_` parts.

        Args:
            c_predicted_noise (torch.Tensor): Predicted noise of shape (B, N, V).
            c_noise (torch.Tensor): GT noise of shape (B, N, V).
            d_log_x0 (torch.Tensor): The log one-hot vector of the initial state of
                shape (B, C, N).
            d_log_xt (torch.Tensor): The log one-hot vector of the noised state of shape
                (B, C, N).
            d_log_x0_recon (torch.Tensor): The reconstructed log one-hot vector of the
                initial state (model output) of shape (B, C, N).
            d_log_pred_prob (torch.Tensor): The log probabilities for the predicted
                classes at timestep `t-1` of shape (B, C, N).
            timesteps (torch.LongTensor): The timesteps of shape (B,).

        Return:
            torch.Tensor: The loss item tensor.
        """
        batch_size = c_predicted_noise.shape[0]

        c_loss = self.continous_loss_function(
            predicted_noise=c_predicted_noise, noise=c_noise
        )
        d_loss = self.discrete_loss_function(
            log_x0=d_log_x0,
            log_xt=d_log_xt,
            log_x0_recon=d_log_x0_recon,
            log_pred_prob=d_log_pred_prob,
            timesteps=timesteps,
        )

        self.log_dict(
            {"training/continous_loss": c_loss, "training/discrete_loss": d_loss},
            batch_size=batch_size,
        )

        combined_loss = c_loss + d_loss
        return combined_loss

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        phase: str = "training",
        use_ema: bool = False,
    ) -> torch.Tensor:
        """
        DDPM forward pass. Normal diffusion training with maximum likelihood objective.
        Returns the loss.
        """
        scenes = batch["scenes"]

        # Sample a timestep for each scene.
        timesteps = (
            torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (scenes.shape[0],),
            )
            .long()
            .to(self.device)
        )

        # Split into continous and discrete part.
        c_scenes = scenes[..., : self.continous_vec_len]  # Shape (B, N, Vc)
        d_scenes = scenes[..., self.continous_vec_len :]  # Shape (B, N, Vd)

        # Add noise to the continous parts.
        c_noise = torch.randn(c_scenes.shape).to(self.device)  # Shape (B, N, Vc)
        c_scenes_noisy = self.noise_scheduler.add_noise(
            c_scenes, c_noise, timesteps
        )  # Shape (B, N, Vc)

        # Add noise to the discrete parts.
        d_log_x0 = self.discrete_diffusion.onehot_to_log_onehot(
            x_onehot=d_scenes
        )  # Shape (B, ndf, N), ndf = num_diffusion_classes
        d_log_xt = self.discrete_diffusion.q_sample(
            log_x_0=d_log_x0, t=timesteps
        )  # Shape (B, ndf, N)
        d_xt = self.discrete_diffusion.log_onehot_to_index(d_log_xt).squeeze(
            1
        )  # Shape (B, N)

        # predicted_noise has shape (B, N, Vc), log_x0_model has shape (B, ndf-1, N).
        c_predicted_noise, d_log_x0_model = self.denoise(
            x_continous=c_scenes_noisy,
            x_discrete=d_xt,
            timesteps=timesteps,
            cond_dict=batch,
            use_ema=use_ema,
        )

        # Reconstruct discrete part.
        d_log_x0_recon = self.discrete_diffusion.log_pred_from_denoise_out(
            d_log_x0_model
        )  # Shape (B, ndf, N)
        d_log_pred_prob = self.discrete_diffusion.q_posterior(
            log_x0_recon=d_log_x0_recon, log_x_t=d_log_xt, t=timesteps
        )  # Shape (B, ndf, N)

        # Compute loss.
        loss = self.loss_function(
            c_predicted_noise=c_predicted_noise,
            c_noise=c_noise,
            d_log_x0=d_log_x0,
            d_log_xt=d_log_xt,
            d_log_x0_recon=d_log_x0_recon,
            d_log_pred_prob=d_log_pred_prob,
            timesteps=timesteps,
        )
        return loss
