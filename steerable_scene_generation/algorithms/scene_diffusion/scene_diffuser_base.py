import json
import logging
import os
import sys

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from pydrake.all import ModelDirectives

from steerable_scene_generation.algorithms.common.base_pytorch_algo import (
    BasePytorchAlgo,
)
from steerable_scene_generation.algorithms.common.lr_schedulers import get_lr_scheduler
from steerable_scene_generation.datasets.scene.scene import SceneDataset
from steerable_scene_generation.utils.caching import conditional_cache
from steerable_scene_generation.utils.file_parsing import get_scene_directives
from steerable_scene_generation.utils.hf_dataset import (
    get_scene_vec_description_from_metadata,
)
from steerable_scene_generation.utils.knn_utils import get_k_closest_training_examples
from steerable_scene_generation.utils.scene_metrics import (
    compute_total_minimum_distances,
    compute_total_scene_penetrations,
)
from steerable_scene_generation.utils.visualization import (
    get_scene_label_image_renders,
    get_scene_renders,
    get_visualized_scene_htmls,
)

from .inference_time_search import mcts_inference_time_search
from .inpainting_helpers import generate_empty_object_inpainting_masks
from .models import get_noise_scheduler
from .postprocessing import apply_forward_simulation, apply_non_penetration_projection
from .scene_distance import (
    compute_distances_to_training_examples,
    compute_min_scene_distances,
)

logger = logging.getLogger(__name__)


class SceneDiffuserBase(BasePytorchAlgo, ABC):
    """
    Abstract base class for diffusion on scene vectors.
    """

    def __init__(self, cfg, dataset: SceneDataset):
        """
        cfg is a DictConfig object defined by
        `configurations/algorithm/scene_diffuser_base.yaml`.
        """
        # Create the scene vector description.
        if cfg.processed_scene_data_path.endswith(".json"):
            metadata_path = cfg.processed_scene_data_path
        else:
            metadata_path = os.path.join(cfg.processed_scene_data_path, "metadata.json")
        metadata_path = os.path.expanduser(metadata_path)

        # Check if the path is a Hugging Face Hub dataset ID
        if "/" in cfg.processed_scene_data_path and not os.path.exists(metadata_path):
            # Load metadata from Hub.
            from huggingface_hub import hf_hub_download

            metadata_path = hf_hub_download(
                repo_id=cfg.processed_scene_data_path,
                filename="metadata.json",
                repo_type="dataset",
                revision="main",
            )
        with open(metadata_path, "r") as metadata_file:
            metadata = json.load(metadata_file)
        self.scene_vec_desc = get_scene_vec_description_from_metadata(
            metadata=metadata,
            static_directive=cfg.static_directive,
            package_names=[p_map.package_name for p_map in cfg.drake_package_maps],
            package_file_paths=[
                p_map.package_file_path for p_map in cfg.drake_package_maps
            ],
        )

        super().__init__(cfg)  # superclass saves cfg as self.cfg and calls _build_model

        self.dataset = dataset

        # Variable for storing the timesteps at which to visualize intermediate scenes.
        self.visualize_intermediate_scene_timesteps = None

    def _setup_virtual_display_if_needed(self) -> None:
        # Setup a virtual display if needed for rendering.
        if (
            sys.platform == "linux"
            and os.getenv("DISPLAY") is None
            and self.global_rank == 0
            and not self.cfg.visualization.use_blender_server
        ):
            logger.info("Setting up virtual display for rendering.")
            from pyvirtualdisplay import Display

            virtual_display = Display(visible=0, size=(1400, 900))
            virtual_display.start()

    def _build_model(self):
        """Create the noise scheduler."""
        self.noise_scheduler = get_noise_scheduler(
            name=self.cfg.noise_schedule.scheduler,
            num_train_timesteps=self.cfg.noise_schedule.num_train_timesteps,
            beta_schedule=self.cfg.noise_schedule.beta_schedule,
        )

    def on_save_checkpoint(self, checkpoint):
        # Save the normalizer state. Loading will happen in the dataset constructor.
        checkpoint["normalizer_state"] = self.dataset.normalizer.get_state()

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[dict[str, Any]]]:
        """Return the optimizer we want to use."""
        parameters = self.parameters()
        optim = torch.optim.AdamW(
            parameters, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )

        num_training_steps = (
            self.trainer.max_epochs
            if self.cfg.lr_scheduler.epochs_as_steps
            else self.trainer.max_steps
        )
        scheduler = get_lr_scheduler(
            optimizer=optim,
            name=self.cfg.lr_scheduler.name,
            num_training_steps=num_training_steps,
            num_warmup_steps=self.cfg.lr_scheduler.num_warmup_steps,
        )
        lr_interval = "epoch" if self.cfg.lr_scheduler.epochs_as_steps else "step"

        return [optim], [{"scheduler": scheduler, "interval": lr_interval}]

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self.cfg.reset_lr_scheduler:
            # Clear the optimizer states to ensure a fresh LR scheduler start.
            if "lr_schedulers" in checkpoint:
                checkpoint["lr_schedulers"] = []
            logger.info("Cleared LR scheduler state.")

        if not self.cfg.ema.use and "state_dict" in checkpoint:
            # Filter out EMA-related keys from the state dict.
            filtered_state_dict = {
                k: v
                for k, v in checkpoint["state_dict"].items()
                if not k.startswith("ema.")
            }
            checkpoint["state_dict"] = filtered_state_dict

            # When loading a checkpoint with EMA but EMA is disabled,
            # we need to reset the optimizer state to avoid size mismatch errors.
            if "optimizer_states" in checkpoint:
                checkpoint["optimizer_states"] = []

            logger.info(
                "Removed EMA-related keys from state dict when loading checkpoint."
            )

    @abstractmethod
    def put_model_in_eval_mode(self) -> None:
        """Put the denoising model in evaluation mode."""
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        phase: str = "training",
        use_ema: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass that returns the loss.
        """
        raise NotImplementedError

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        See BasePytorchAlgo class for detailed documentation.
        Args:
            batch: The output of your data iterable, normally a
                :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.

        Return:
            A loss tensor or loss dictionary.
        """
        # Prevent logging ambiguitiy.
        batch_size = batch["scenes"].shape[0]

        loss = self.forward(batch)
        self.log_dict({"training/loss": loss}, batch_size=batch_size)

        if self.cfg.ema.use and self.cfg.ema.log_at_train:
            ema_loss = self.forward(batch, phase="validation", use_ema=True)
            self.log_dict({"training/ema_loss": ema_loss}, batch_size=batch_size)

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # Prevent logging ambiguitiy.
        batch_size = batch["scenes"].shape[0]

        loss = self.forward(batch, phase="validation")
        self.log_dict({"validation/loss": loss}, sync_dist=True, batch_size=batch_size)

        use_ema = self.cfg.ema.use
        if use_ema:
            ema_loss = self.forward(batch, phase="validation", use_ema=True)
            self.log_dict(
                {"validation/ema_loss": ema_loss}, sync_dist=True, batch_size=batch_size
            )

        if batch_idx == 0 and self.global_rank == 0:
            self._setup_virtual_display_if_needed()

            data_batch = batch[0] if isinstance(batch, list) else batch
            if self.cfg.validation.num_samples_to_render > 0:
                images = self.get_scene_renders(
                    num_scenes=self.cfg.validation.num_samples_to_render,
                    batch_size=self.cfg.validation.sample_batch_size,
                    data_batch=data_batch,
                )
                self.log_image("validation/scene_renders", images)

                if use_ema:
                    ema_images = self.get_scene_renders(
                        num_scenes=self.cfg.validation.num_samples_to_render,
                        batch_size=self.cfg.validation.sample_batch_size,
                        use_ema=True,
                        data_batch=data_batch,
                    )
                    self.log_image("validation/ema_scene_renders", ema_images)

            if self.cfg.validation.num_samples_to_visualize > 0:
                htmls = self.get_scene_htmls(
                    num_scenes=self.cfg.validation.num_samples_to_visualize,
                    batch_size=self.cfg.validation.sample_batch_size,
                    data_batch=data_batch,
                )
                self.log_html("validation/scene_htmls", htmls)

            if self.cfg.validation.num_directives_to_generate > 0:
                scene_directives = self.get_scene_directives(
                    num_scenes=self.cfg.validation.num_directives_to_generate,
                    batch_size=self.cfg.validation.sample_batch_size,
                    data_batch=data_batch,
                )
                self.log_drake_directives("drake_scene_directives", scene_directives)

        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # Prevent logging ambiguitiy.
        batch_size = batch["scenes"].shape[0]

        loss = self.forward(batch, phase="test")
        self.log_dict({"test/loss": loss}, batch_size=batch_size)

        use_ema = self.cfg.ema.use and self.cfg.test.use_ema
        if use_ema:
            ema_loss = self.forward(batch, phase="test", use_ema=True)
            self.log_dict(
                {"test/ema_loss": ema_loss}, sync_dist=True, batch_size=batch_size
            )

        if batch_idx == 0 and self.global_rank == 0:
            self._setup_virtual_display_if_needed()

            data_batch = batch[0] if isinstance(batch, list) else batch
            if self.cfg.test.num_samples_to_render > 0:
                images = self.get_scene_renders(
                    num_scenes=self.cfg.test.num_samples_to_render,
                    is_test=True,
                    batch_size=self.cfg.test.sample_batch_size,
                    use_ema=use_ema,
                    data_batch=data_batch,
                )
                self.log_image("test/scene_renders", images)

            if self.cfg.test.num_samples_to_render_as_label > 0:
                images = self.get_scene_label_renders(
                    num_scenes=self.cfg.test.num_samples_to_render_as_label,
                    is_test=True,
                    batch_size=self.cfg.test.sample_batch_size,
                    use_ema=use_ema,
                    data_batch=data_batch,
                )
                self.log_image("test/scene_label_renders", images)

            if self.cfg.test.num_samples_to_visualize > 0:
                htmls = self.get_scene_htmls(
                    num_scenes=self.cfg.test.num_samples_to_visualize,
                    is_test=True,
                    batch_size=self.cfg.test.sample_batch_size,
                    use_ema=use_ema,
                    data_batch=data_batch,
                )
                self.log_html("test/scene_htmls", htmls)

            if self.cfg.test.num_directives_to_generate > 0:
                scene_directives = self.get_scene_directives(
                    num_scenes=self.cfg.test.num_directives_to_generate,
                    is_test=True,
                    batch_size=self.cfg.test.sample_batch_size,
                    use_ema=use_ema,
                    data_batch=data_batch,
                )
                self.log_drake_directives("drake_scene_directives", scene_directives)

            if self.cfg.test.num_samples_to_save_as_pickle > 0:
                scene_dict = self.get_scenes_as_dict_for_saving(
                    num_scenes=self.cfg.test.num_samples_to_save_as_pickle,
                    is_test=True,
                    batch_size=self.cfg.test.sample_batch_size,
                    use_ema=use_ema,
                    data_batch=data_batch,
                )
                self.log_pickle("scene_dict", scene_dict)

        return loss

    def on_predict_start(self):
        # Set different seed for each worker to prevent sample duplicates.
        new_seed = torch.initial_seed() + self.global_rank
        torch.manual_seed(new_seed)
        torch.cuda.manual_seed_all(new_seed)

        # Initialize empty lists to store all predictions and indices.
        self.all_predictions = []
        self.all_scene_indices = []

    def predict_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor | dict[str, Any]:
        if (
            sum(
                [
                    self.cfg.predict.do_sample,
                    self.cfg.predict.do_rearrange,
                    self.cfg.predict.do_complete,
                    self.cfg.predict.do_inference_time_search,
                    self.cfg.predict.do_sample_scenes_with_k_closest_training_examples,
                ]
            )
            != 1
        ):
            raise ValueError("Need to specify exactly one prediction mode.")

        use_ema = self.cfg.ema.use and self.cfg.test.use_ema

        if self.cfg.predict.do_sample:
            num_samples = len(batch["scenes"])
            # Returns scenes of shape (B, N, V).
            predictions = self.sample_scenes(
                num_samples=num_samples,
                is_test=True,
                use_ema=use_ema,
                data_batch=batch,
            )
        elif self.cfg.predict.do_rearrange:
            # Returns scenes of shape (B, N, V).
            predictions = self.rearrange_scenes(data_batch=batch, use_ema=use_ema)
        elif self.cfg.predict.do_complete:
            # Returns scenes of shape (B, N, V).
            predictions = self.complete_scenes(data_batch=batch, use_ema=use_ema)
        elif self.cfg.predict.do_inference_time_search:
            # Returns a dict.
            predictions = self.inference_time_search(data_batch=batch, use_ema=use_ema)
        elif self.cfg.predict.do_sample_scenes_with_k_closest_training_examples:
            # Returns a dict.
            predictions = self.sample_scenes_with_k_closest_training_examples(
                data_batch=batch, use_ema=use_ema
            )

        # Append to all_predictions list.
        self.all_predictions.append(predictions)
        if "scene_indices" in batch:
            self.all_scene_indices.append(batch["scene_indices"])

        return predictions

    def on_predict_end(self):
        if self.cfg.predict.do_inference_time_search:
            # Search isn't currently supported in distributed mode.
            return

        # Concatenate predictions from all epochs.
        if isinstance(self.all_predictions[0], torch.Tensor):
            self.predictions = torch.cat(self.all_predictions, dim=0)
        elif isinstance(self.all_predictions[0], dict):
            self.predictions = {
                k: torch.cat([p[k] for p in self.all_predictions], dim=0)
                for k in self.all_predictions[0].keys()
            }
        else:
            raise ValueError(
                f"Invalid prediction type: {type(self.all_predictions[0])}"
            )

        # Concatenate scene indices if they exist.
        if self.all_scene_indices:
            self.prediction_scene_indices = torch.cat(self.all_scene_indices, dim=0)
        else:
            self.prediction_scene_indices = None

        # Gather predictions and annotations from all workers.
        gathered_predictions = self.all_gather(self.predictions)  # Shape (W, B, ...)

        # Skip shape correction if used a single worker.
        if isinstance(gathered_predictions, torch.Tensor):
            if gathered_predictions.dim() == self.predictions.dim():
                return
        elif isinstance(gathered_predictions, dict):
            # Check any tensor in the dictionary to determine if we need to reshape.
            for key, value in gathered_predictions.items():
                if isinstance(value, torch.Tensor):
                    if value.dim() == self.predictions[key].dim():
                        return
                    break
        else:
            raise ValueError(f"Invalid prediction type: {type(gathered_predictions)}")

        # Flatten predictions along world_size dimension.
        if isinstance(gathered_predictions, torch.Tensor):
            gathered_predictions = (
                gathered_predictions.detach()
                .cpu()
                .reshape(-1, *gathered_predictions.shape[2:])
            )  # Shape (B, N, V)
        elif isinstance(gathered_predictions, dict):
            # Flatten each tensor in the dictionary.
            for key, value in gathered_predictions.items():
                if not isinstance(value, torch.Tensor):
                    raise ValueError(
                        f"Invalid prediction dict value type: {type(value)}"
                    )
                gathered_predictions[key] = (
                    value.detach().cpu().reshape(-1, *value.shape[2:])
                )
        else:
            raise ValueError(f"Invalid prediction type: {type(gathered_predictions)}")

        if self.prediction_scene_indices is not None:
            gathered_scene_indices = (
                self.all_gather(self.prediction_scene_indices).detach().cpu()
            )
            gathered_scene_indices = gathered_scene_indices.reshape(-1)

            # Sort predictions by scene indices.
            sorted_indices = torch.argsort(gathered_scene_indices)
            if isinstance(gathered_predictions, torch.Tensor):
                self.predictions = gathered_predictions[sorted_indices]
            elif isinstance(gathered_predictions, dict):
                for key, value in gathered_predictions.items():
                    self.predictions[key] = value[sorted_indices]
        else:
            # Store predictions as is.
            self.predictions = gathered_predictions

    def visualize_intermediate_scene(self, t: int, xt: torch.Tensor) -> None:
        """
        Visualize an intermediate scene if the current timestep is in the list of
        timesteps to visualize.

        Args:
            t (int): The current timestep.
            xt (torch.Tensor): The normalized scene to visualize. Shape (B, N, V).
        """
        if self.visualize_intermediate_scene_timesteps is None:
            # Determine timesteps at which to visualize intermediate scenes.
            total_timesteps = len(self.noise_scheduler.timesteps)
            num_visualize_timesteps = (
                self.cfg.visualization.num_intermediate_scenes_to_visualize
            )
            visualize_timesteps = [
                int(t * total_timesteps / (num_visualize_timesteps))
                for t in range(num_visualize_timesteps + 1)
            ]
            self.visualize_intermediate_scene_timesteps = visualize_timesteps

        if t not in self.visualize_intermediate_scene_timesteps:
            # Skip this timestep.
            return

        # Apply inverse normalization.
        xt_unnorm = self.dataset.inverse_normalize_scenes(xt)  # Shape (B, N, V)

        renders = self.get_scene_renders(num_scenes=1, scenes=xt_unnorm)
        self.log_image(f"intermediate_scene/t_{t}", renders)

    def get_scene_renders(
        self,
        num_scenes: int,
        is_test: bool = False,
        scenes: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        use_ema: bool = False,
        data_batch: Dict[str, torch.Tensor] | None = None,
    ) -> List[torch.Tensor]:
        """
        Return rendered RGBA images of the scenes. One image is returned per scene.

        Args:
            num_scenes (int): The number of scenes to visualize.
            is_test (bool): Whether its the testing phase.
            scenes (Optional[torch.Tensor]): The unormalized scenes to visualize. If
                None, scenes will be sampled from the model.
            batch_size (Optional[int]): The batch size to use for sampling scenes. If
                None, sample all scenes at once.
            use_ema (bool): Whether to use the EMA model.
            data_batch (Dict[str, torch.Tensor] | None): The optional data batch. Some
                sampling methods might use parts of this as conditioning for sampling.

        Returns:
            List[torch.Tensor]: The rendered RGBA images.
        """
        if scenes is None:
            sampled_scenes = self.sample_scenes(
                num_samples=num_scenes,
                is_test=is_test,
                batch_size=batch_size,
                use_ema=use_ema,
                data_batch=data_batch,
            )
        else:
            sampled_scenes = scenes[:num_scenes]
        images = get_scene_renders(
            scenes=sampled_scenes,
            scene_vec_desc=self.scene_vec_desc,
            camera_poses=self.cfg.visualization.camera_pose,
            camera_width=self.cfg.visualization.image_width,
            camera_height=self.cfg.visualization.image_height,
            background_color=self.cfg.visualization.background_color,
            num_workers=self.cfg.visualization.num_workers,
            use_blender_server=self.cfg.visualization.use_blender_server,
            blender_server_url=self.cfg.visualization.blender_server_url,
        )
        return torch.tensor(images)

    def get_scene_label_renders(
        self,
        num_scenes: int,
        is_test: bool = False,
        scenes: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        use_ema: bool = False,
        data_batch: Dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Return rendered RGB label images of the scenes. Each model has a unique color.
        One image is returned per scene.

        Args:
            num_scenes (int): The number of scenes to visualize.
            is_test (bool): Whether its the testing phase.
            scenes (Optional[torch.Tensor]): The unormalized scenes to visualize. If
                None, scenes will be sampled from the model.
            batch_size (Optional[int]): The batch size to use for sampling scenes. If
                None, sample all scenes at once.
            use_ema (bool): Whether to use the EMA model.
            data_batch (Dict[str, torch.Tensor] | None): The optional data batch. Some
                sampling methods might use parts of this as conditioning for sampling.

        Returns:
            torch.Tensor: The rendered images.
        """
        if scenes is None:
            sampled_scenes = self.sample_scenes(
                num_samples=num_scenes,
                is_test=is_test,
                batch_size=batch_size,
                use_ema=use_ema,
                data_batch=data_batch,
            )
        else:
            sampled_scenes = scenes[:num_scenes]
        images = get_scene_label_image_renders(
            scenes=sampled_scenes,
            scene_vec_desc=self.scene_vec_desc,
            camera_poses=self.cfg.visualization.camera_pose,
            camera_width=self.cfg.visualization.image_width,
            camera_height=self.cfg.visualization.image_height,
            num_workers=self.cfg.visualization.num_workers,
        )
        return torch.tensor(images)

    def get_scene_htmls(
        self,
        num_scenes: int,
        is_test: bool = False,
        scenes: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        use_ema: bool = False,
        data_batch: Dict[str, torch.Tensor] | None = None,
    ) -> List[str]:
        """
        Return HTML visualizations of the scenes.

        Args:
            num_scenes (int): The number of scenes to visualize.
            is_test (bool): Whether its the testing phase.
            scenes (Optional[torch.Tensor]): The unormalized scenes to visualize. If
                None, scenes will be sampled from the model.
            batch_size (Optional[int]): The batch size to use for sampling scenes. If
                None, sample all scenes at once.
            use_ema (bool): Whether to use the EMA model.
            data_batch (Dict[str, torch.Tensor] | None): The optional data batch. Some
                sampling methods might use parts of this as conditioning for sampling.
        """
        if scenes is None:
            sampled_scenes = self.sample_scenes(
                num_samples=num_scenes,
                is_test=is_test,
                batch_size=batch_size,
                use_ema=use_ema,
                data_batch=data_batch,
            )
        else:
            sampled_scenes = scenes[:num_scenes]
        htmls = get_visualized_scene_htmls(
            scenes=sampled_scenes,
            scene_vec_desc=self.scene_vec_desc,
            visualize_proximity=self.cfg.visualization.visualize_proximity,
            weld_objects=self.cfg.visualization.weld_objects,
            simulation_time=3.0 if self.cfg.visualization.weld_objects else 1e-6,
            background_color=self.cfg.visualization.background_color,
        )
        return htmls

    def get_scene_directives(
        self,
        num_scenes: int,
        is_test: bool = False,
        scenes: torch.Tensor | None = None,
        batch_size: int | None = None,
        use_ema: bool = False,
        data_batch: dict[str, torch.Tensor] | None = None,
    ) -> list[ModelDirectives]:
        """
        Samples scenes and returns Drake directives for them:
        https://drake.mit.edu/doxygen_cxx/structdrake_1_1multibody_1_1parsing_1_1_model_directives.html

        Args:
            num_scenes (int): The number of scenes to visualize.
            is_test (bool): Whether its the testing phase.
            scenes (torch.Tensor | None): The unormalized scenes to visualize. If
                None, scenes will be sampled from the model.
            batch_size (int | None): The batch size to use for sampling scenes. If
                None, sample all scenes at once.
            use_ema (bool): Whether to use the EMA model.
            data_batch (dict[str, torch.Tensor] | None): The optional data batch. Some
                sampling methods might use parts of this as conditioning for sampling.

        Returns:
            list[ModelDirectives]: The Drake directives for the sampled scenes of shape
                (B,).
        """
        if scenes is None:
            sampled_scenes = self.sample_scenes(
                num_samples=num_scenes,
                is_test=is_test,
                batch_size=batch_size,
                use_ema=use_ema,
                data_batch=data_batch,
            )
        else:
            sampled_scenes = scenes[:num_scenes]
        directives = get_scene_directives(
            scenes=sampled_scenes, scene_vec_desc=self.scene_vec_desc
        )
        return directives

    def get_scenes_as_dict_for_saving(
        self,
        num_scenes: int,
        is_test: bool = False,
        scenes: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        use_ema: bool = False,
        data_batch: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, Any]:
        """
        Get the scenes in a dictionary format that can be saved to disk.

        Args:
            num_scenes (int): The number of scenes to visualize.
            is_test (bool): Whether its the testing phase.
            scenes (Optional[torch.Tensor]): The unormalized scenes to visualize. If
                None, scenes will be sampled from the model.
            batch_size (Optional[int]): The batch size to use for sampling scenes. If
                None, sample all scenes at once.
            use_ema (bool): Whether to use the EMA model.
            data_batch (Dict[str, torch.Tensor] | None): The optional data batch. Some
                sampling methods might use parts of this as conditioning for sampling.

        Returns:
            Dict[str, Any]: The scenes in a dictionary format.
        """
        if scenes is None:
            sampled_scenes = self.sample_scenes(
                num_samples=num_scenes,
                is_test=is_test,
                batch_size=batch_size,
                use_ema=use_ema,
                data_batch=data_batch,
            )
        else:
            sampled_scenes = scenes[:num_scenes]

        sampled_scenes = sampled_scenes.detach()
        sampled_scenes_normalized_np = (
            self.dataset.normalize_scenes(sampled_scenes).cpu().numpy()
        )
        sampled_scenes_np = sampled_scenes.cpu().numpy()

        scene_dict = {
            "scenes": sampled_scenes_np,
            "scenes_normalized": sampled_scenes_normalized_np,
            "scene_vec_desc": self.scene_vec_desc,
        }
        return scene_dict

    def compute_physical_feasibility_metrics(
        self,
        num_scenes: int,
        name: str,
        is_test: bool = False,
        scenes: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        use_ema: bool = False,
        data_batch: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, float]:
        if scenes is None:
            sampled_scenes = self.sample_scenes(
                num_samples=num_scenes,
                is_test=is_test,
                batch_size=batch_size,
                use_ema=use_ema,
                data_batch=data_batch,
            )
        else:
            sampled_scenes = scenes[:num_scenes]

        total_scene_penetrations = compute_total_scene_penetrations(
            scenes=sampled_scenes,
            scene_vec_desc=self.scene_vec_desc,
            num_workers=self.cfg.sample_metrics.num_workers,
        )
        median_total_scene_penetrations = total_scene_penetrations.median()

        total_minimum_distances = compute_total_minimum_distances(
            scenes=sampled_scenes,
            scene_vec_desc=self.scene_vec_desc,
            num_workers=self.cfg.sample_metrics.num_workers,
        )
        median_total_minimum_distances = total_minimum_distances.median()

        prefix = "ema_" if use_ema else ""
        return {
            f"{name}/{prefix}median_total_scene_penetrations": median_total_scene_penetrations,
            f"{name}/{prefix}median_total_minimum_distances": median_total_minimum_distances,
        }

    def apply_postprocessing(self, scenes: torch.Tensor) -> torch.Tensor:
        """
        Apply post-processing to the scenes.

        Args:
            scenes (torch.Tensor): The scenes to post-process. Shape (B, N, V). The
                scenes are unormalized.

        Returns:
            torch.Tensor: The post-processed scenes. Shape (B, N, V). The scenes are
                unormalized. The returned scenes are detached and on the CPU.
        """
        # Store original scenes to use as fallback for failed processing.
        original_scenes = scenes.clone()
        caches = [None for _ in range(len(scenes))]
        if self.cfg.postprocessing.apply_non_penetration_projection:
            scenes, caches, successes = apply_non_penetration_projection(
                scenes=scenes,
                scene_vec_desc=self.scene_vec_desc,
                translation_only=self.cfg.postprocessing.non_penetration_projection.translation_only,
                influence_distance=self.cfg.postprocessing.non_penetration_projection.influence_distance,
                solver_name=self.cfg.postprocessing.non_penetration_projection.solver_name,
                iteration_limit=self.cfg.postprocessing.non_penetration_projection.iteration_limit,
                caches=caches,
                num_workers=self.cfg.postprocessing.num_workers,
            )

            if (
                self.cfg.postprocessing.non_penetration_projection.discard_failed_projection_scenes
                and not self.cfg.postprocessing.return_original_scenes_on_failure
            ):
                successfull_scenes = scenes[successes]
                logger.info(
                    f"Projection succeeded for {sum(successes)}/{len(scenes)} scenes."
                )
                if len(successfull_scenes) == 0:
                    logger.info(
                        "All scenes failed to project, returning original scenes."
                    )
                    return original_scenes
                scenes = successfull_scenes
                caches = [cache for cache, success in zip(caches, successes) if success]
                successes = [True] * len(scenes)
            elif not all(successes):
                # Replace failed scenes with original scenes.
                logger.info(
                    f"Projection succeeded for {sum(successes)}/{len(scenes)} scenes. "
                    f"Replacing failed scenes with original scenes."
                )
                for i, success in enumerate(successes):
                    if not success:
                        scenes[i] = original_scenes[i]
        else:
            successes = [True] * len(scenes)

        if self.cfg.postprocessing.apply_forward_simulation:
            # Only simulate successful scenes to prevent errors from simulating scenes
            # that start in deep penetration.
            scenes_to_simulate = scenes[successes]
            caches = [cache for cache, success in zip(caches, successes) if success]
            simulated_scenes, _ = apply_forward_simulation(
                scenes=scenes_to_simulate,
                scene_vec_desc=self.scene_vec_desc,
                simulation_time_s=self.cfg.postprocessing.forward_simulation.simulation_time_s,
                time_step=self.cfg.postprocessing.forward_simulation.time_step,
                timeout_s=self.cfg.postprocessing.forward_simulation.timeout_s,
                caches=caches,
                num_workers=self.cfg.postprocessing.num_workers,
            )
            # Add non-simulated scenes back to the list in the original order.
            simulated_scenes_idx = 0
            for i, success in enumerate(successes):
                if success:
                    scenes[i] = simulated_scenes[simulated_scenes_idx]
                    simulated_scenes_idx += 1

            # Handle NaN values that might occur during simulation.
            nan_scenes = torch.isnan(scenes).any(dim=-1).any(dim=-1)
            if torch.sum(nan_scenes) > 0:
                if self.cfg.postprocessing.return_original_scenes_on_failure:
                    # Replace scenes with NaN values with original scenes.
                    logger.warning(
                        f"Replacing {nan_scenes.sum().item()} scenes containing NaN "
                        "values with original scenes."
                    )
                    scenes[nan_scenes] = original_scenes[nan_scenes]
                elif torch.sum(nan_scenes) == len(scenes):
                    logger.warning(
                        "All scenes have NaN values, returning original scenes."
                    )
                    return original_scenes
                else:
                    # Remove scenes with NaN values.
                    scenes = scenes[~nan_scenes]
                    num_scenes_removed = nan_scenes.sum().item()
                    logger.warning(
                        f"Removed {num_scenes_removed} scenes with NaN values!"
                    )

        return scenes

    @torch.no_grad()
    def log_sampled_scene_metrics(
        self, sampled_scenes: torch.Tensor, name: str
    ) -> None:
        """
        Computes and logs sampled scene metrics.

        Args:
            sampled_scenes (torch.Tensor): The sampled scenes of shape (B, N, V). The
                scenes are unormalized.
            name (str): The name to use for logging the metrics.
        """
        if self.cfg.sample_metrics.compute_scene_distance_between_samples:
            # Get normalized scenes.
            sampled_scenes_norm = self.dataset.normalize_scenes(sampled_scenes)

            # Compute minimum distances between samples.
            (
                min_distances_between_samples,
                pairwise_distances_between_samples,
            ) = compute_min_scene_distances(
                sampled_scenes_norm,
                batch_size=self.cfg.sample_metrics.batch_size,
                return_pairwise_distances=True,
            )
            mean_min_distance_between_samples = torch.mean(
                min_distances_between_samples
            )
            # Create a mask to exclude self-comparisons (diagonal elements).
            mask = ~torch.eye(
                pairwise_distances_between_samples.shape[0],
                dtype=torch.bool,
                device=pairwise_distances_between_samples.device,
            )
            mean_distance_between_samples = (
                pairwise_distances_between_samples[mask].mean().item()
            )

            # Compute number of duplicates.
            mask = (
                pairwise_distances_between_samples
                < self.cfg.sample_metrics.duplicate_distance_theshold
            )
            upper_triangle_indices = torch.triu_indices(
                pairwise_distances_between_samples.size(0),
                pairwise_distances_between_samples.size(1),
                offset=1,  # Exclude the diagonal
            )
            duplicate_count = torch.sum(
                mask[upper_triangle_indices[0], upper_triangle_indices[1]]
            ).item()

            self.log_dict(
                {
                    f"{name}/mean_distance_between_samples": mean_distance_between_samples,
                    f"{name}/mean_min_distance_between_samples": mean_min_distance_between_samples,
                    f"{name}/duplicate_sample_count": duplicate_count,
                }
            )

        if self.cfg.sample_metrics.compute_scene_penetration:
            total_scene_penetrations = compute_total_scene_penetrations(
                scenes=sampled_scenes,
                scene_vec_desc=self.scene_vec_desc,
                num_workers=self.cfg.sample_metrics.num_workers,
            )
            mean_total_scene_penetrations = total_scene_penetrations.mean()
            median_total_scene_penetrations = total_scene_penetrations.median()

            total_minimum_distances = compute_total_minimum_distances(
                scenes=sampled_scenes,
                scene_vec_desc=self.scene_vec_desc,
                num_workers=self.cfg.sample_metrics.num_workers,
            )
            median_total_minimum_distances = total_minimum_distances.median()

            self.log_dict(
                {
                    f"{name}/mean_total_scene_penetrations": mean_total_scene_penetrations,
                    f"{name}/median_total_scene_penetrations": median_total_scene_penetrations,
                    f"{name}/median_total_minimum_distances": median_total_minimum_distances,
                }
            )

    @abstractmethod
    @conditional_cache(argument_name="is_test")  # Only cache during test time.
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
        raise NotImplementedError

    @abstractmethod
    def inpaint_scenes(
        self, data_batch: Dict[str, torch.Tensor], use_ema: bool = False
    ) -> torch.Tensor:
        """
        Inpaints the scenes in the data batch by regenerating everything inside the
        inpainting masks while keeping everything outside the masks fixed.

        Args:
            data_batch (Dict[str, torch.Tensor]): The data batch with normalized scenes
                of shape (B, N, V) and boolean inpainting masks of shape (B, N, V).
                These can be used to inpaint the continuous part, discrete part, a
                combination of both, or even a subset of the continuous part. Note that
                it is not possible to inpaint a subset of the discrete part.
            use_ema (bool, optional): Whether to use the EMA model. Defaults to False.

        Returns:
            torch.Tensor: The inpainted, unnormalized scenes of shape (B, N, V).
        """
        raise NotImplementedError

    def rearrange_scenes(
        self, data_batch: Dict[str, torch.Tensor], use_ema: bool = False
    ) -> torch.Tensor:
        """
        Rearrange the scenes in the data batch by regenerating the continuous part while
        keeping the discrete part fixed.

        Args:
            data_batch (Dict[str, torch.Tensor]): The data batch with normalized scenes
                of shape (B, N, V).
            use_ema (bool, optional): Whether to use the EMA model. Defaults to False.

        Returns:
            torch.Tensor: The rearranged, unnormalized scenes of shape (B, N, V).
        """
        scenes = data_batch["scenes"]  # Shape (B, N, V)

        # Create inpainting masks that only mask the continuous part.
        continuous_part_len = (
            self.scene_vec_desc.get_object_vec_len()
            - self.scene_vec_desc.model_path_vec_len
        )
        inpainting_masks = torch.zeros_like(scenes, dtype=torch.bool)  # Shape (B, N, V)
        inpainting_masks[..., :continuous_part_len] = True

        inpaint_data_batch = data_batch.copy()
        inpaint_data_batch["inpainting_masks"] = inpainting_masks

        # Inpaint the continuous part.
        rearranged_scenes = self.inpaint_scenes(inpaint_data_batch, use_ema=use_ema)

        return rearranged_scenes

    def complete_scenes(
        self, data_batch: Dict[str, torch.Tensor], use_ema: bool = False
    ) -> torch.Tensor:
        """
        Completes the scenes in the data batch by regenerating the empty objects
        while keeping the non-empty objects fixed. Both discrete and continuous parts
        are regenerated for the empty objects. Note that this might result in an empty
        object staying empty.

        Args:
            data_batch (Dict[str, torch.Tensor]): The data batch with normalized scenes
                of shape (B, N, V).
            use_ema (bool, optional): Whether to use the EMA model. Defaults to False.

        Returns:
            torch.Tensor: The completed, unnormalized scenes of shape (B, N, V).
        """
        scenes = data_batch["scenes"]  # Shape (B, N, V)

        # Create inpainting masks that only mask empty objects.
        inpainting_masks, _ = generate_empty_object_inpainting_masks(
            scenes=scenes, scene_vec_desc=self.scene_vec_desc
        )

        inpaint_data_batch = data_batch.copy()
        inpaint_data_batch["inpainting_masks"] = inpainting_masks

        # Inpaint the empty objects.
        completed_scenes = self.inpaint_scenes(inpaint_data_batch, use_ema=use_ema)

        return completed_scenes

    def inference_time_search(
        self, data_batch: Dict[str, torch.Tensor] | None = None, use_ema: bool = False
    ) -> dict[str, Any]:
        """
        Samples a single scene using greedy inference-time search.

        Args:
            data_batch (Dict[str, torch.Tensor] | None): The data batch containing
                scenes of shape (B, N, V) or None to create a dummy batch.
            use_ema (bool, optional): Whether to use the Exponential Moving Average
                (EMA) model. Defaults to False.

        Returns:
            dict[str, Any]: A dictionary containing the following keys:
                - "reached_max_iters": Whether search terminated because the maximum
                    number of steps was reached.
                - "num_iters_used": The number of steps used.
                - "history": The history of scenes generated of shape (T, N, V). The
                    scenes are unnormalized. The first scene is the initial scene and
                    the last scene is the final scene. Scenes should monotonically
                    improve due to the greedy search.
                - "best_scenes": The best scenes encountered during the search. All
                    problematic objects have been removed. This scene is unnormalized.
                    Shape (K, N, V).
                - "best_scene_indices": The indices of the best scenes in the history.
                - "best_costs": The costs of the best scenes in the history.
                - "total_penetration_distances": The total penetration distances of the
                    scenes in the history. Only present if using the non-penetration
                    objective.
                - "empty_object_numbers": The number of empty objects in the history.
                    Only present if using the object number objective.
                - "non_empty_object_numbers": The number of non-empty objects in the
                    history. Only present if using the object number objective.
                - "tree_data": The data for the MCTS tree. Only present if using MCTS.
        """
        if data_batch is None:
            # Create a dummy data batch.
            inpaint_data_batch = {
                "scenes": torch.zeros(
                    1,
                    self.cfg.max_num_objects_per_scene,
                    self.scene_vec_desc.get_object_vec_len(),
                )
            }
        else:
            inpaint_data_batch = data_batch.copy()

        # Optionally replace the data labels with the specified labels during testing.
        if not self.cfg.classifier_free_guidance.sampling.use_data_labels:
            txt_labels = self.cfg.classifier_free_guidance.sampling.labels
            inpaint_data_batch = self.dataset.replace_cond_data(
                data=inpaint_data_batch, txt_labels=txt_labels
            )

        num_objects_per_scene = (
            self.cfg.max_num_objects_per_scene
            + self.cfg.num_additional_tokens_for_sampling
        )
        return mcts_inference_time_search(
            inpaint_data_batch=inpaint_data_batch,
            max_num_objects_per_scene=num_objects_per_scene,
            use_ema=use_ema,
            cfg=self.cfg.predict.inference_time_search,
            scene_vec_desc=self.scene_vec_desc,
            dataset=self.dataset,
            inpaint_function=self.inpaint_scenes,
        )

    def sample_continuous_noise_prior(
        self, scenes_shape: tuple[int, ...] | torch.Size
    ) -> torch.Tensor:
        """
        Sample the continuous noise from prior distribution p(xT) (which normaly is a
        standard Gaussian).

        Args:
          scenes_shape: (B, N, V)

        Returns:
            torch.Tensor: The continuous noise of shape (B, N, V).
        """
        noise = torch.randn(scenes_shape).to(self.device)
        return noise

    def sample_scenes_with_k_closest_training_examples(
        self, data_batch: Dict[str, torch.Tensor], use_ema: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Sample scenes and find the k closest training examples.

        Args:
            data_batch (Dict[str, torch.Tensor]): A dictionary containing the input
                data for sampling scenes.
            use_ema (bool): Whether to use the Exponential Moving Average (EMA) model
                for sampling.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the following keys:
                - "sampled_scenes": The sampled scenes of shape (B, N, V).
                - "closest_training_scenes": The closest training scenes of shape
                    (B, K, N, V).
                - "distances": The distances to the closest examples of shape (B, K).
        """
        scenes = self.sample_scenes(
            num_samples=len(data_batch["scenes"]),
            is_test=True,
            use_ema=use_ema,
            data_batch=data_batch,
        )

        closest_training_scenes, distances = get_k_closest_training_examples(
            scenes=scenes,
            dataset=self.dataset,
            num_k=self.cfg.predict.sample_scenes_with_k_closest_training_examples.num_k,
            batch_size=self.cfg.predict.sample_scenes_with_k_closest_training_examples.batch_size,
        )

        return {
            "sampled_scenes": scenes,
            "closest_training_scenes": closest_training_scenes,
            "distances": distances,
        }
