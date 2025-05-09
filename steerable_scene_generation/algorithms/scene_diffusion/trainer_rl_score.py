from typing import Dict

import torch

from steerable_scene_generation.datasets.scene.scene import SceneDataset

from .trainer_rl import SceneDiffuserTrainerRL


class SceneDiffuserTrainerScore(SceneDiffuserTrainerRL):
    """
    Class that provides REINFORCE (score function gradient estimator) training logic.
    This corresponds to DPPO_{SF} (https://arxiv.org/abs/2305.13301).
    """

    def __init__(self, cfg, dataset: SceneDataset):
        """
        cfg is a DictConfig object defined by
        `configurations/algorithm/scene_diffuser_base_continous.yaml`.
        """
        super().__init__(cfg, dataset=dataset)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        phase: str = "training",
        use_ema: bool = False,
    ) -> torch.Tensor:
        """
        DDPO-like (https://arxiv.org/abs/2305.13301) forward pass. Training with RL.
        Returns the loss.
        """

        # Get diffusion trajectories.
        (
            trajectories,  # Shape (B, T+1, N, V)
            trajectories_log_props,  # Shape (B, T)
            cond_dict,
        ) = self.generate_trajs_for_ddpo(
            last_n_timesteps_only=self.cfg.ddpo.last_n_timesteps_only,
            n_timesteps_to_sample=self.cfg.ddpo.n_timesteps_to_sample,
            batch=batch,
        )

        # Remove initial noisy scene.
        trajectories = trajectories[:, 1:]  # Shape (B, T, N, V)

        # Compute rewards.
        rewards = self.compute_rewards_from_trajs(
            trajectories=trajectories, cond_dict=cond_dict
        )  # Shape (B,)

        # Compute advantages.
        advantages = self.compute_advantages(rewards, phase=phase)  # Shape (B,)

        # REINFORCE loss.
        loss = -torch.mean(torch.sum(trajectories_log_props, dim=1) * advantages)

        # DDPM loss for regularization.
        if self.cfg.ddpo.ddpm_reg_weight > 0.0:
            loss += self.compute_ddpm_loss(batch) * self.cfg.ddpo.ddpm_reg_weight

        return loss
