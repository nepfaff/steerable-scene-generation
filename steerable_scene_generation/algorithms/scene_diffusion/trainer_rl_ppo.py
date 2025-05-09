from typing import Dict

import torch

from diffusers import DDIMScheduler, DDPMScheduler
from tqdm import tqdm

from steerable_scene_generation.datasets.scene.scene import SceneDataset

from .ddpo_helpers import ddim_step_with_logprob, ddpm_step_with_logprob
from .trainer_rl import SceneDiffuserTrainerRL


class SceneDiffuserTrainerPPO(SceneDiffuserTrainerRL):
    """
    Class that provides PPO-like (https://arxiv.org/abs/1707.06347) training logic.
    Note that there is no value function. This corresponds to DPPO_{IS}
    (https://arxiv.org/abs/2305.13301).

    Note that this requires `DDPStrategy(find_unused_parameters=True)` for distributed
    training.
    """

    def __init__(self, cfg, dataset: SceneDataset):
        """
        cfg is a DictConfig object defined by
        `configurations/algorithm/scene_diffuser_base_continous.yaml`.
        """
        super().__init__(cfg, dataset=dataset)

        # Variables to store collected trajectory data to enable multiple optimization
        # steps on the same data batch.
        self.samples = None
        self.prev_next_samples = None
        self.trajectories_log_props = None
        self.advantages = None
        self.timesteps = None

    def recompute_trajectory_data(
        self, batch: Dict[str, torch.Tensor], phase: str = "training"
    ) -> None:
        """Re-compute trajectory data."""
        with torch.no_grad():
            # Get diffusion trajectories.
            (
                trajectories,  # Shape (B, T+1, N, V)
                trajectories_log_props,  # Shape (B, T)
                cond_dict,
            ) = self.generate_trajs_for_ddpo(batch=batch)

            # Compute rewards.
            rewards = self.compute_rewards_from_trajs(
                trajectories=trajectories, cond_dict=cond_dict
            )  # Shape (B,)

            # Compute advantages.
            advantages = self.compute_advantages(rewards, phase=phase)  # Shape (B,)

            # Reshape so that time dimension comes first.
            trajectories = trajectories.transpose(0, 1)  # Shape (T+1, B, N, V)
            trajectories_log_props = trajectories_log_props.transpose(
                0, 1
            )  # Shape (T, B)

            # Need to keep the same initial states for both the original and new policy.
            samples = trajectories[:-1]  # Shape (T, B, N, V)
            prev_next_samples = trajectories[1:]  # Shape (T, B, N, V)

            if self.cfg.ddpo.last_n_timesteps_only != 0:
                # Only keep the last timesteps.
                timesteps = self.noise_scheduler.timesteps[
                    -self.cfg.ddpo.last_n_timesteps_only :
                ]
                samples = samples[-self.cfg.ddpo.last_n_timesteps_only :]
                prev_next_samples = prev_next_samples[
                    -self.cfg.ddpo.last_n_timesteps_only :
                ]
                trajectories_log_props = trajectories_log_props[
                    -self.cfg.ddpo.last_n_timesteps_only :
                ]
                advantages = advantages[-self.cfg.ddpo.last_n_timesteps_only :]
            else:
                # Keep all timesteps.
                timesteps = self.noise_scheduler.timesteps  # Shape (T,)

            # Save data.
            self.samples = samples
            self.prev_next_samples = prev_next_samples
            self.trajectories_log_props = trajectories_log_props
            self.advantages = advantages
            self.timesteps = timesteps

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

        if isinstance(self.noise_scheduler, DDIMScheduler):
            self.noise_scheduler.set_timesteps(
                self.cfg.noise_schedule.ddim.num_inference_timesteps,
                device=self.device,
            )

        # Re-compute trajectory data at beginning of each epoch.
        if self.samples is None:
            self.recompute_trajectory_data(batch=batch, phase=phase)

        loss = 0.0

        # Determine which timesteps to use for gradient computation.
        timestep_indices = list(range(len(self.timesteps)))
        if self.cfg.ddpo.n_timesteps_to_sample > 0:
            # Sample specific timesteps for gradient computation.
            n_to_sample = min(
                self.cfg.ddpo.n_timesteps_to_sample, len(timestep_indices)
            )
            timestep_indices = sorted(
                torch.randperm(len(timestep_indices))[:n_to_sample].tolist()
            )

        # Only iterate over selected timesteps.
        for idx in tqdm(
            timestep_indices,
            total=len(timestep_indices),
            desc="  Sampling scenes (PPO)",
            leave=False,
            position=1,
        ):
            t = self.timesteps[idx]
            sample = self.samples[idx]  # Shape (B, N, V)
            prev_sample = self.prev_next_samples[idx]  # Shape (B, N, V)
            old_log_prop = self.trajectories_log_props[idx]  # Shape (B,)

            # Use the same advantage for all timesteps in the trajectory.
            advantage = self.advantages  # Shape (B,)

            residual = self.predict_noise(sample, t)  # Shape (B, N, V)

            # Compute the log probability.
            if isinstance(self.noise_scheduler, DDPMScheduler):
                _, log_prop = ddpm_step_with_logprob(
                    scheduler=self.noise_scheduler,
                    model_output=residual,
                    timestep=t,
                    sample=sample,
                    prev_sample=prev_sample,
                )
            else:  # DDIMScheduler
                _, log_prop = ddim_step_with_logprob(
                    scheduler=self.noise_scheduler,
                    model_output=residual,
                    timestep=t,
                    sample=sample,
                    prev_sample=prev_sample,
                    eta=self.cfg.noise_schedule.ddim.eta,
                )

            # Ratio between action (prev_sample) likelihood under new and old policy
            # (denoising model), given the current state (sample).
            ratio = torch.exp(log_prop - old_log_prop)  # Shape (B,)

            unclipped_loss = -advantage * ratio  # Shape (B,)
            clipped_loss = -advantage * torch.clamp(
                ratio,
                min=1.0 - self.cfg.ddpo.ppo.clip_range,
                max=1.0 + self.cfg.ddpo.ppo.clip_range,
            )  # Shape (B,)
            loss += torch.mean(torch.maximum(unclipped_loss, clipped_loss))

            # Log clip fraction.
            clip_fraction = (
                (torch.abs(ratio - 1.0) > self.cfg.ddpo.ppo.clip_range).float().mean()
            )
            self.log(
                f"{phase}/clip_fraction", clip_fraction, on_step=False, on_epoch=True
            )

        # DDPM loss for regularization.
        if self.cfg.ddpo.ddpm_reg_weight > 0.0:
            loss += self.compute_ddpm_loss(batch) * self.cfg.ddpo.ddpm_reg_weight

        return loss

    def optimizer_step(self, *args, **kwargs):
        """
        Run multiple iterations of gradient descent on the same data batch.
        """
        for _ in range(self.cfg.ddpo.ppo.num_epochs):
            super().optimizer_step(*args, **kwargs)

        # Reset data batch.
        self.samples = None
        self.prev_next_samples = None
        self.trajectories_log_props = None
        self.advantages = None
        self.timesteps = None
