import pathlib

from typing import Optional, Union

from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig

from .exp_base import BaseExperiment
from .scene_diffusion import SceneDiffusionExperiment

# each key has to be a yaml file under '[project_root]/configurations/experiment'
# without .yaml suffix
exp_registry = dict(
    scene_diffusion=SceneDiffusionExperiment,
)


def build_experiment(
    cfg: DictConfig,
    logger: Optional[WandbLogger] = None,
    ckpt_path: Optional[Union[str, pathlib.Path]] = None,
) -> BaseExperiment:
    """
    Build an experiment instance based on registry
    :param cfg: configuration file
    :param logger: optional logger for the experiment
    :param ckpt_path: optional checkpoint path for saving and loading
    :return:
    """
    if cfg.experiment._name not in exp_registry:
        raise ValueError(
            f"Experiment {cfg.experiment._name} not found in registry "
            f"{list(exp_registry.keys())}. Make sure you register it correctly in "
            "'experiments/__init__.py' under the same name as yaml file."
        )

    return exp_registry[cfg.experiment._name](cfg, logger, ckpt_path)
