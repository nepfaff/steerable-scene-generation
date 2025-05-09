from typing import Dict, Optional, Tuple

import torch

from diffusers import DDIMScheduler, DDPMScheduler
from tqdm import tqdm

from steerable_scene_generation.datasets.scene.scene import SceneDataset

from .ddpo_helpers import (
    ddim_step_with_logprob,
    ddpm_step_with_logprob,
    non_penetration_reward,
    number_of_physically_feasible_objects_reward,
    object_number_reward,
    prompt_following_reward,
)
from .scene_diffuser_base_continous import SceneDiffuserBaseContinous
from .trainer_ddpm import compute_ddpm_loss


class SceneDiffuserTrainerRL(SceneDiffuserBaseContinous):
    """
    Base class for classes that provide RL training logic.
    """

    def __init__(self, cfg, dataset: SceneDataset):
        """
        cfg is a DictConfig object defined by
        `configurations/algorithm/scene_diffuser_base_continous.yaml`.
        """
        super().__init__(cfg, dataset=dataset)

        # Variable for storing the reward computation cache.
        self.reward_cache = None

    def generate_trajs_for_ddpo(
        self,
        last_n_timesteps_only: int = 0,
        n_timesteps_to_sample: int = 0,
        batch: Dict[str, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict | None]:
        """
        Generate denoising trajectories for DDPO.

        Args:
            last_n_timesteps_only (int): If not 0, only keep the last n timesteps.
            n_timesteps_to_sample (int): If not 0, uniformly sample this many timesteps
                for gradient computation. All other timesteps will use torch.no_grad().
            batch (Dict[str, torch.Tensor] | None): Training batch to sample
                conditioning from.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, dict]: A batch of denoising trajectories
            (shape: (B, N+1, V)), their log probabilities (shape: (B,)) and
            corresponding conditioning dictionary.
        """
        assert self.cfg.ddpo.batch_size > 1, "Need at least 2 samples for DDPO."
        assert not (
            last_n_timesteps_only != 0 and n_timesteps_to_sample != 0
        ), "Cannot specify both last_n_timesteps_only and n_timesteps_to_sample"

        self.put_model_in_eval_mode()

        if isinstance(self.noise_scheduler, DDIMScheduler):
            self.noise_scheduler.set_timesteps(
                self.cfg.noise_schedule.ddim.num_inference_timesteps, device=self.device
            )

        # Determine which timestep indices to compute gradients for.
        timesteps_with_grads = set()
        if n_timesteps_to_sample > 0:
            # Uniformly sample timestep indices.
            timesteps_with_grads = set(
                range(len(self.noise_scheduler.timesteps))
                if n_timesteps_to_sample >= len(self.noise_scheduler.timesteps)
                else torch.randperm(len(self.noise_scheduler.timesteps))[
                    :n_timesteps_to_sample
                ].tolist()
            )
        elif last_n_timesteps_only > 0:
            # Use the last n timesteps.
            num_timesteps = len(self.noise_scheduler.timesteps)
            last_n = min(last_n_timesteps_only, num_timesteps)
            timesteps_with_grads = set(range(num_timesteps - last_n, num_timesteps))
        else:
            # Use all timesteps.
            timesteps_with_grads = set(range(len(self.noise_scheduler.timesteps)))

        trajectory = []
        trajectory_log_props = []

        # Sample random noise.
        num_objects_per_scene = (
            self.cfg.max_num_objects_per_scene
            + self.cfg.num_additional_tokens_for_sampling
        )
        xt = self.sample_continuous_noise_prior(
            (
                self.cfg.ddpo.batch_size,
                num_objects_per_scene,
                self.scene_vec_desc.get_object_vec_len(),
            )
        ).to(
            self.device
        )  # Shape (B, N, V)
        trajectory.append(xt)

        # Create conditioning dictionary from batch if available.
        cond_dict = None
        if batch is not None:
            cond_dict = self.dataset.sample_data_dict(
                data=batch, num_items=self.cfg.ddpo.batch_size
            )

        for t_idx, t in enumerate(
            tqdm(
                self.noise_scheduler.timesteps,
                desc="  Sampling scenes (Traj generation)",
                leave=False,
                position=1,
            )
        ):
            # Predict the noise for the current timestep.
            if t_idx not in timesteps_with_grads:
                # Don't compute gradients.
                with torch.no_grad():
                    residual = self.predict_noise(xt, t, cond_dict=cond_dict)
            else:
                residual = self.predict_noise(
                    xt, t, cond_dict=cond_dict
                )  # Shape (B, N, V)

            # Compute the updated sample and log probability.
            if isinstance(self.noise_scheduler, DDPMScheduler):
                xt, log_prop = ddpm_step_with_logprob(
                    scheduler=self.noise_scheduler,
                    model_output=residual,
                    timestep=t,
                    sample=xt,
                )
            else:  # DDIMScheduler
                xt, log_prop = ddim_step_with_logprob(
                    scheduler=self.noise_scheduler,
                    model_output=residual,
                    timestep=t,
                    sample=xt,
                    eta=self.cfg.noise_schedule.ddim.eta,
                )

            trajectory.append(xt)
            trajectory_log_props.append(log_prop)

        # Stack so that batch dimension is first.
        trajectories = torch.stack(trajectory, dim=1)  # Shape (B, T+1, N, V)
        trajectories_log_props = torch.stack(
            trajectory_log_props, dim=1
        )  # Shape (B, T)

        if last_n_timesteps_only != 0:
            trajectories = torch.cat(
                (trajectories[:, :1], trajectories[:, -last_n_timesteps_only:]), dim=1
            )
            trajectories_log_props = trajectories_log_props[:, -last_n_timesteps_only:]

        return trajectories, trajectories_log_props, cond_dict

    def compute_rewards_from_trajs(
        self,
        trajectories: torch.Tensor,
        cond_dict: dict | None = None,
        are_trajectories_normalized: bool = True,
    ) -> torch.Tensor:
        """
        Compute rewards from denoising trajectories.

        Args:
            trajectories (torch.Tensor): Denoising trajectories of shape (B, T, N, V).
            cond_dict (dict | None): Conditioning dictionary that was used to generate
                the trajectories.
            are_trajectories_normalized (bool): Whether the trajectories are normalized.

        Returns:
            torch.Tensor: Rewards of shape (B,)
        """
        if (
            sum(
                [
                    self.cfg.ddpo.use_non_penetration_reward,
                    self.cfg.ddpo.use_object_number_reward,
                    self.cfg.ddpo.use_prompt_following_reward,
                    self.cfg.ddpo.use_physical_feasible_objects_reward,
                ]
            )
            > 1
        ):
            raise ValueError("Only one reward function is supported at a time.")

        # Only compute rewards for the last timestep.
        x0 = trajectories[:, -1]  # Shape (B, N, V)

        if are_trajectories_normalized:
            # Apply inverse normalization.
            x0 = self.dataset.inverse_normalize_scenes(x0)  # Shape (B, N, V)

        if self.cfg.ddpo.use_non_penetration_reward:
            rewards, self.reward_cache = non_penetration_reward(
                scenes=x0,
                scene_vec_desc=self.scene_vec_desc,
                num_workers=self.cfg.ddpo.num_reward_workers,
                cache=self.reward_cache,
                return_updated_cache=True,
            )
        elif self.cfg.ddpo.use_object_number_reward:
            rewards = object_number_reward(
                scenes=x0, scene_vec_desc=self.scene_vec_desc
            )
        elif self.cfg.ddpo.use_prompt_following_reward:
            prompts = cond_dict["language_annotation"]
            rewards = prompt_following_reward(
                scenes=x0, prompts=prompts, scene_vec_desc=self.scene_vec_desc
            )
        elif self.cfg.ddpo.use_physical_feasible_objects_reward:
            rewards = number_of_physically_feasible_objects_reward(
                scenes=x0,
                scene_vec_desc=self.scene_vec_desc,
                cfg=self.cfg.ddpo.physical_feasibility,
                num_workers=self.cfg.ddpo.num_reward_workers,
            )
        else:
            raise ValueError("Need to select one reward function.")

        return rewards

    def compute_advantages(
        self, rewards: torch.Tensor, phase: str = "training"
    ) -> torch.Tensor:
        """
        Compute advantages from rewards. The advantages are normalized rewards.

        When using multiple GPU workers, this method synchronizes reward statistics
        across all workers to ensure consistent advantage scaling.

        Args:
            rewards (torch.Tensor): Rewards of shape (B,).
            phase (str): Phase of training. Used for logging.

        Returns:
            torch.Tensor: Advantages of shape (B,).
        """
        # Small epsilon to prevent division by zero.
        eps = 1e-12

        # Synchronize statistics across all workers if using distributed training.
        if self.trainer.world_size > 1:
            # Compute local statistics.
            local_reward_mean = rewards.mean()
            local_reward_squared_mean = (rewards**2).mean()

            # Gather statistics from all workers.
            gathered_means = self.all_gather(local_reward_mean)
            gathered_squared_means = self.all_gather(local_reward_squared_mean)

            # Compute global statistics.
            reward_mean = gathered_means.mean()
            # Need to aggregate according to Var = E[(X - μ)²] = E[X²] - μ².
            global_reward_squared_mean = gathered_squared_means.mean()
            global_reward_var = global_reward_squared_mean - reward_mean**2
            reward_std = torch.sqrt(torch.clamp(global_reward_var, min=eps))
        else:
            # Use local statistics for single worker.
            reward_mean = rewards.mean()
            reward_std = rewards.std()

        # Compute advantages using synchronized statistics.
        advantages = (rewards - reward_mean) / (reward_std + eps)  # Shape (B,)

        self.log_dict(
            {
                f"{phase}/mean_reward": reward_mean.item(),
                f"{phase}/std_reward": reward_std.item(),
            },
            sync_dist=True,
            batch_size=self.cfg.ddpo.batch_size,
        )

        # Clip the advantages
        advantages = torch.clamp(
            advantages,
            min=-self.cfg.ddpo.advantage_max,
            max=self.cfg.ddpo.advantage_max,
        )

        return advantages

    def compute_ddpm_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        DDPM forward pass.
        This is a replication of the DDPM forward pass in the `trainer_ddpm.py` file
        to allow the RL trainers to stay separate from the DDPM trainer.
        """
        scenes = batch["scenes"]

        # Sample noise to add to the scenes.
        noise = torch.randn(scenes.shape).to(self.device)  # Shape (B, N, V)

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

        # Add noise to the scenes.
        noisy_scenes = self.noise_scheduler.add_noise(
            scenes, noise, timesteps
        )  # Shape (B, N, V)

        predicted_noise = self.predict_noise(
            noisy_scenes=noisy_scenes, timesteps=timesteps, cond_dict=batch
        )  # Shape (B, N, V)

        # Compute loss.
        loss = compute_ddpm_loss(
            predicted_noise=predicted_noise,
            noise=noise,
            scene_vec_desc=self.scene_vec_desc,
            cfg=self.cfg,
            log_fn=self.log_dict,
        )

        return loss

    def sample_scenes(
        self,
        num_samples: int,
        is_test: bool = False,
        batch_size: Optional[int] = None,
        use_ema: bool = False,
        data_batch: Dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Sample scenes from the model. The returned scenes are unormalized.

        Args:
            num_samples (int): The number of scenes to sample.
            is_test (bool): Whether its the testing phase.
            batch_size (Optional[int]): The batch size to use for sampling scenes. If
                None, sample all scenes at once.
            use_ema (bool): Whether to use the EMA model.
            data_batch (Dict[str, torch.Tensor] | None): The optional data batch. Some
                sampling methods might use parts of this as conditioning for sampling.

        Returns:
            torch.Tensor: The unnormalized scenes of shape (num_samples, N, V).
        """
        cond_dict = None
        if data_batch is not None:
            cond_dict = self.dataset.sample_data_dict(
                data=data_batch, num_items=num_samples
            )

        sampled_scenes = super().sample_scenes(
            num_samples=num_samples,
            is_test=is_test,
            batch_size=batch_size,
            use_ema=use_ema,
            data_batch=cond_dict,
        )

        # Compute rewards for the sampled scenes.
        with torch.no_grad():
            # Predict doesn't support logging.
            if not self.trainer.state.stage == "predict":
                rewards = self.compute_rewards_from_trajs(
                    sampled_scenes.unsqueeze(1),  # Shape (B, 1, N, V)
                    cond_dict=cond_dict,
                    are_trajectories_normalized=False,
                )
                self.log("sampled_scenes/reward", rewards.mean().item())

        return sampled_scenes
