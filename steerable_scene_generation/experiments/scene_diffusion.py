from steerable_scene_generation.algorithms.scene_diffusion import (
    SceneDiffuserTrainerDDPM,
    SceneDiffuserTrainerMixedDiffusion,
    SceneDiffuserTrainerPPO,
    SceneDiffuserTrainerScore,
    create_scene_diffuser_diffuscene,
    create_scene_diffuser_flux_transformer,
    create_scene_diffuser_midiffusion,
    create_scene_diffuser_mixed_diffuscene,
    create_scene_diffuser_mixed_flux_transformer,
    create_scene_diffuser_mixed_midiffusion,
)
from steerable_scene_generation.datasets import SceneDataset

from .exp_base import BaseLightningExperiment


class SceneDiffusionExperiment(BaseLightningExperiment):
    """A scene diffusion experiment."""

    # each key has to be a yaml file under '[project_root]/configurations/algorithm'
    # without .yaml suffix
    compatible_algorithm_factories = dict(
        # Continous diffusion.
        scene_diffuser_flux_transformer=create_scene_diffuser_flux_transformer,
        scene_diffuser_diffuscene=create_scene_diffuser_diffuscene,
        scene_diffuser_midiffusion=create_scene_diffuser_midiffusion,
        # Mixed diffusion.
        scene_diffuser_mixed_flux_transformer=create_scene_diffuser_mixed_flux_transformer,
        scene_diffuser_mixed_diffuscene=create_scene_diffuser_mixed_diffuscene,
        scene_diffuser_mixed_midiffusion=create_scene_diffuser_mixed_midiffusion,
    )

    compatible_algorithm_trainers = dict(
        # Continous diffusion.
        ddpm=SceneDiffuserTrainerDDPM,
        rl_score=SceneDiffuserTrainerScore,
        rl_ppo=SceneDiffuserTrainerPPO,
        # Mixed diffusion.
        mixed=SceneDiffuserTrainerMixedDiffusion,
    )

    # each key has to be a yaml file under '[project_root]/configurations/dataset'
    # without .yaml suffix
    compatible_datasets = dict(
        scene=SceneDataset,
    )

    def _build_algo(self, ckpt_path=None):
        """
        Build the lightning module
        :return:  a pytorch-lightning module to be launched
        """
        algo_name = self.root_cfg.algorithm._name
        if algo_name not in self.compatible_algorithm_factories:
            raise ValueError(
                f"Algorithm {algo_name} not found in compatible_algorithm_factories for "
                "this Experiment class. Make sure you define "
                "compatible_algorithm_factories correctly  and make sure that each key "
                "has same name as yaml file under "
                "'[project_root]/configurations/algorithm' without .yaml suffix."
            )

        trainer_name = self.root_cfg.algorithm.trainer
        if trainer_name not in self.compatible_algorithm_trainers:
            raise ValueError(
                f"Trainer {trainer_name} not found in compatible_algorithm_trainers "
                "for  this Experiment class. Make sure you define "
                "compatible_algorithm_trainers correctly."
            )

        # Construct dataset for normalization.
        dataset = self._build_dataset("training", ckpt_path=ckpt_path)

        # Construct algorithm.
        algo_factory_func = self.compatible_algorithm_factories[algo_name]
        trainer = self.compatible_algorithm_trainers[trainer_name]
        algo = algo_factory_func(trainer)(self.root_cfg.algorithm, dataset)
        return algo
