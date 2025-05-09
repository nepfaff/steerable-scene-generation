import os
import pathlib
import random

from abc import ABC
from datetime import timedelta
from typing import Any, Dict, Optional, Union

import hydra
import lightning.pytorch as pl
import numpy as np
import torch

from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from omegaconf import DictConfig

from steerable_scene_generation.datasets.common import BaseDataset
from steerable_scene_generation.utils.distributed_utils import is_rank_zero
from steerable_scene_generation.utils.print_utils import cyan

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class BaseExperiment(ABC):
    """
    Abstract class for an experiment. This generalizes the pytorch lightning Trainer &
    lightning Module to more flexible experiments that doesn't fit in the typical ml
    loop, e.g. multi-stage reinforcement learning benchmarks.
    """

    # each key has to be a yaml file under '[project_root]/configurations/algorithm'
    # without .yaml suffix
    compatible_algorithms: Dict = NotImplementedError

    def __init__(
        self,
        root_cfg: DictConfig,
        logger: Optional[WandbLogger] = None,
        ckpt_path: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        """
        Constructor

        Args:
            cfg: configuration file that contains everything about the experiment
            logger: a pytorch-lightning WandbLogger instance
            ckpt_path: an optional path to saved checkpoint
        """
        super().__init__()
        self.root_cfg = root_cfg
        self.cfg = root_cfg.experiment
        self.debug = root_cfg.debug
        self.logger = logger
        self.ckpt_path = ckpt_path
        self.algo = None

        # Set seed.
        base_seed = self.cfg.seed if self.cfg.seed is not None else 42
        if self.logger is not None:
            self.logger.log_hyperparams({"seed": base_seed})

        # Get worker rank.
        rank = 0
        if hasattr(self, "global_rank"):
            rank = self.global_rank
        elif torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()

        # Use different seed for each worker.
        worker_seed = base_seed + rank
        torch.manual_seed(worker_seed)
        torch.cuda.manual_seed_all(worker_seed)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        seed_everything(worker_seed, workers=True)
        if self.cfg.seed is not None:
            # Note that this slows down training.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True

        if self.cfg.matmul_precision:
            torch.set_float32_matmul_precision(self.cfg.matmul_precision)

    def _build_algo(self, ckpt_path=None):
        """
        Build the lightning module
        :return:  a pytorch-lightning module to be launched
        """
        algo_name = self.root_cfg.algorithm._name
        if algo_name not in self.compatible_algorithms:
            raise ValueError(
                f"Algorithm {algo_name} not found in compatible_algorithms for this "
                "Experiment class. Make sure you define compatible_algorithms correctly "
                "and make sure that each key has same name as yaml file under "
                "'[project_root]/configurations/algorithm' without .yaml suffix."
            )
        return self.compatible_algorithms[algo_name](self.root_cfg.algorithm)

    def exec_task(self, task: str, **kwargs) -> Any:
        """
        Executing a certain task specified by string. Each task should be a stage of
        experiment. In most computer vision / nlp applications, tasks should be just
        train and test. In reinforcement learning, you might have more stages such as
        collecting dataset etc

        Args:
            task: a string specifying a task implemented for this experiment
        """

        if hasattr(self, task) and callable(getattr(self, task)):
            if is_rank_zero:
                print(cyan("Executing task:"), f"{task} out of {self.cfg.tasks}")
            return getattr(self, task)(**kwargs)
        else:
            raise ValueError(
                f"Specified task '{task}' not defined for class "
                f"{self.__class__.__name__} or is not callable."
            )


class BaseLightningExperiment(BaseExperiment):
    """
    Abstract class for pytorch lightning experiments. Useful for computer vision & nlp
    where main components are simply models, datasets and train loop.
    """

    # each key has to be a yaml file under '[project_root]/configurations/algorithm'
    # without .yaml suffix
    compatible_algorithms: Dict = NotImplementedError

    # each key has to be a yaml file under '[project_root]/configurations/dataset'
    # without .yaml suffix
    compatible_datasets: Dict = NotImplementedError

    def _build_trainer_callbacks(self):
        callbacks = []
        if self.logger:
            callbacks.append(LearningRateMonitor("step", True))

    def _build_training_loader(
        self, ckpt_path: Optional[str] = None
    ) -> Optional[Union[TRAIN_DATALOADERS, pl.LightningDataModule]]:
        train_dataset = self._build_dataset("training", ckpt_path)
        if not train_dataset:
            return None

        sampler = train_dataset.get_sampler()
        shuffle = (
            False
            if (
                isinstance(train_dataset, torch.utils.data.IterableDataset)
                or sampler is not None
            )
            else self.cfg.test.data.shuffle
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            sampler=sampler,
            batch_size=self.cfg.training.batch_size,
            num_workers=min(os.cpu_count(), self.cfg.training.data.num_workers),
            shuffle=shuffle,
            persistent_workers=True,
            pin_memory=self.cfg.training.pin_memory,
            prefetch_factor=self.cfg.training.prefetch_factor,
        )

    def _build_validation_loader(
        self, ckpt_path: Optional[str] = None
    ) -> Optional[Union[TRAIN_DATALOADERS, pl.LightningDataModule]]:
        validation_dataset = self._build_dataset("validation", ckpt_path)
        if not validation_dataset:
            return

        sampler = validation_dataset.get_sampler()
        shuffle = (
            False
            if (
                isinstance(validation_dataset, torch.utils.data.IterableDataset)
                or sampler is not None
            )
            else self.cfg.test.data.shuffle
        )
        return torch.utils.data.DataLoader(
            validation_dataset,
            sampler=sampler,
            batch_size=self.cfg.validation.batch_size,
            num_workers=min(os.cpu_count(), self.cfg.validation.data.num_workers),
            shuffle=shuffle,
            persistent_workers=True,
            pin_memory=self.cfg.validation.pin_memory,
            prefetch_factor=self.cfg.validation.prefetch_factor,
        )

    def _build_test_loader(
        self, ckpt_path: Optional[str] = None
    ) -> Optional[Union[TRAIN_DATALOADERS, pl.LightningDataModule]]:
        test_dataset = self._build_dataset("test", ckpt_path)
        if not test_dataset:
            return None

        sampler = test_dataset.get_sampler()
        shuffle = (
            False
            if (
                isinstance(test_dataset, torch.utils.data.IterableDataset)
                or sampler is not None
            )
            else self.cfg.test.data.shuffle
        )
        return torch.utils.data.DataLoader(
            test_dataset,
            sampler=sampler,
            batch_size=self.cfg.test.batch_size,
            num_workers=min(os.cpu_count(), self.cfg.test.data.num_workers),
            shuffle=shuffle,
            persistent_workers=True,
            pin_memory=self.cfg.test.pin_memory,
            prefetch_factor=self.cfg.test.prefetch_factor,
        )

    def training(self) -> None:
        """
        All training happens here
        """
        if not self.algo:
            self.algo = self._build_algo(ckpt_path=self.ckpt_path)
        if self.cfg.training.compile:
            self.algo = torch.compile(self.algo)

        callbacks = []
        if self.logger:
            callbacks.append(LearningRateMonitor("step", True))
        if "checkpointing" in self.cfg.training:
            callbacks.append(
                ModelCheckpoint(
                    pathlib.Path(
                        hydra.core.hydra_config.HydraConfig.get()["runtime"][
                            "output_dir"
                        ]
                    )
                    / "checkpoints",
                    **self.cfg.training.checkpointing,
                )
            )

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            devices="auto",
            strategy=(
                DDPStrategy(
                    process_group_backend="nccl",
                    find_unused_parameters=self.cfg.find_unused_parameters,
                    timeout=timedelta(seconds=3600),
                )
                if torch.cuda.device_count() > 1
                else "auto"
            ),
            callbacks=callbacks,
            gradient_clip_val=self.cfg.training.optim.gradient_clip_val,
            val_check_interval=self.cfg.validation.val_every_n_step,
            limit_val_batches=self.cfg.validation.limit_batch,
            check_val_every_n_epoch=self.cfg.validation.val_every_n_epoch,
            accumulate_grad_batches=self.cfg.training.optim.accumulate_grad_batches,
            precision=self.cfg.training.precision,
            detect_anomaly=self.cfg.debug,
            num_sanity_val_steps=0,  # int(self.cfg.debug),
            max_epochs=self.cfg.training.max_epochs,
            max_steps=self.cfg.training.max_steps,
            max_time=self.cfg.training.max_time,
            log_every_n_steps=self.cfg.training.log_every_n_steps,
            fast_dev_run=self.cfg.training.fast_dev_run,
        )

        if self.cfg.debug and self.logger is not None:
            self.logger.watch(self.algo, log="all")

        if self.cfg.training.load_weights_only and self.ckpt_path is not None:
            ckpt = torch.load(self.ckpt_path)
            self.algo.load_state_dict(ckpt["state_dict"])

            trainer.fit(
                self.algo,
                train_dataloaders=self._build_training_loader(self.ckpt_path),
                val_dataloaders=self._build_validation_loader(self.ckpt_path),
                ckpt_path=None,
            )
        else:
            trainer.fit(
                self.algo,
                train_dataloaders=self._build_training_loader(self.ckpt_path),
                val_dataloaders=self._build_validation_loader(self.ckpt_path),
                ckpt_path=self.ckpt_path,
            )

    def validation(self) -> None:
        """
        All validation happens here
        """
        if not self.algo:
            self.algo = self._build_algo(ckpt_path=self.ckpt_path)
        if self.cfg.validation.compile:
            self.algo = torch.compile(self.algo)

        callbacks = []

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            devices=1,
            callbacks=callbacks,
            limit_val_batches=self.cfg.validation.limit_batch,
            precision=self.cfg.validation.precision,
            detect_anomaly=False,  # self.cfg.debug,
        )

        # if self.debug and self.logger is not None:
        #     self.logger.watch(self.algo, log="all")

        trainer.validate(
            self.algo,
            dataloaders=self._build_validation_loader(self.ckpt_path),
            ckpt_path=self.ckpt_path,
        )

    def test(self) -> None:
        """
        All testing happens here
        """
        if not self.algo:
            self.algo = self._build_algo(ckpt_path=self.ckpt_path)
        if self.cfg.test.compile:
            self.algo = torch.compile(self.algo)

        callbacks = []

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            devices=1,
            callbacks=callbacks,
            limit_test_batches=self.cfg.test.limit_batch,
            precision=self.cfg.test.precision,
            detect_anomaly=False,  # self.cfg.debug,
            inference_mode=self.cfg.test.inference_mode,
        )

        # Only load the checkpoint if only testing. Otherwise, it will have been loaded
        # and further trained during train.
        trainer.test(
            self.algo,
            dataloaders=self._build_test_loader(self.ckpt_path),
            ckpt_path=self.ckpt_path,
        )

    def predict(
        self,
        dataloader: Optional[torch.utils.data.DataLoader] = None,
        callbacks: Optional[list] = None,
    ) -> torch.Tensor:
        """Used for scaling inference to multiple devices."""
        if not self.algo:
            self.algo = self._build_algo(ckpt_path=self.ckpt_path)
        if self.cfg.test.compile:
            self.algo = torch.compile(self.algo)

        callbacks = [] if callbacks is None else callbacks

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            devices="auto",
            strategy=(
                DDPStrategy(
                    process_group_backend="nccl",
                    find_unused_parameters=self.cfg.find_unused_parameters,
                    timeout=timedelta(seconds=3600),
                )
                if torch.cuda.device_count() > 1
                else "auto"
            ),
            callbacks=callbacks,
            precision=self.cfg.training.precision,
            inference_mode=self.cfg.test.inference_mode,
        )

        predictions = trainer.predict(
            self.algo,
            dataloaders=(
                self._build_test_loader(self.ckpt_path)
                if dataloader is None
                else dataloader
            ),
            ckpt_path=self.ckpt_path,
            return_predictions=True,
        )

        return predictions

    def _build_dataset(
        self, split: str, ckpt_path: Optional[str] = None
    ) -> Optional[BaseDataset]:
        if split in ["training", "test", "validation"]:
            return self.compatible_datasets[self.root_cfg.dataset._name](
                self.root_cfg.dataset, split=split, ckpt_path=ckpt_path
            )
        else:
            raise NotImplementedError(f"split '{split}' is not implemented")
