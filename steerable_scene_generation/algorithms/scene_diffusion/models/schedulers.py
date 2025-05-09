from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.schedulers.scheduling_utils import SchedulerMixin


def get_noise_scheduler(
    name: str,
    num_train_timesteps: int,
    beta_schedule: str = "linear",
) -> SchedulerMixin:
    """Get the noise scheduler by name.

    Args:
        name: The name of the diffusion scheduler. Should be one of "ddpm" or "ddim".
        num_train_timesteps: The number of training timesteps.
        beta_schedule: The schedule for beta.

    Returns:
        The noise scheduler.
    """
    if name == "ddpm":
        return DDPMScheduler(
            num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule
        )
    elif name == "ddim":
        return DDIMScheduler(
            num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule
        )
    else:
        raise ValueError(f"Invalid diffusion scheduler name: {name}")
