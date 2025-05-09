from typing import Any, Dict, Type, Union

import torch

from steerable_scene_generation.algorithms.common.ema_model import EMAModel
from steerable_scene_generation.algorithms.common.txt_encoding import (
    load_txt_encoder_from_config,
)
from steerable_scene_generation.datasets.scene.scene import SceneDataset

from .models import MIDiffusionContinuous
from .scene_diffuser_base_continous import SceneDiffuserBaseContinous


def create_scene_diffuser_midiffusion(
    trainer_class: Type[SceneDiffuserBaseContinous],
) -> Type[SceneDiffuserBaseContinous]:
    """
    Factory function to create a scene diffuser MIDiffusion class.
    https://arxiv.org/abs/2405.21066

    Args:
        trainer_class (Type[SceneDiffuserBaseContinous]): The base class for the scene
            diffuser. This class should be a subclass of SceneDiffuserBaseContinous and
            implement the forward method and any additional methods needed for the
            training method.

    Returns:
        SceneDiffuser: The scene diffuser class. This class implements any
            model-specific logic needed for the scene diffusion algorithm. See the
            docstring of the returned class for more details.
    """

    class SceneDiffuserMiDiffusion(trainer_class):
        """
        Scene diffusion on a set of un-ordered objects. The number of objects and types
        of objects are not fixed. The object vectors consist of [translation, rotation,
        model_vector]. All scenes have `max_num_objects_per_scene` objects.

        This implements the continuous baseline model from MiDiffusion:
        https://arxiv.org/abs/2405.21066
        """

        def __init__(self, cfg, dataset: SceneDataset):
            """
            cfg is a DictConfig object defined by
            `configurations/algorithm/scene_diffuser_midiffusion.yaml`.
            """
            super().__init__(cfg, dataset=dataset)

        def _build_model(self):
            """Create all pytorch models."""
            super()._build_model()

            # Conditioning.
            if self.cfg.classifier_free_guidance.use:
                self.txt_encoder, text_cond_dim = load_txt_encoder_from_config(
                    self.cfg, component="encoder"
                )
            else:
                self.txt_encoder = None
                text_cond_dim = 0

            network_dim = {
                "objectness_dim": 0,  # Not used by our scene representation
                "class_dim": self.scene_vec_desc.get_model_path_vec_len(),
                "translation_dim": self.scene_vec_desc.get_translation_vec_len(),
                "size_dim": 0,  # Not used by our scene representation
                "angle_dim": self.scene_vec_desc.get_rotation_vec_len(),
                "objfeat_dim": 0,  # Not used by our scene representation
            }
            self.model = MIDiffusionContinuous(
                network_dim=network_dim,
                seperate_all=self.cfg.model.seperate_all,
                n_layer=self.cfg.model.n_layer,
                n_embd=self.cfg.model.n_embd,
                n_head=self.cfg.model.n_head,
                dim_feedforward=self.cfg.model.dim_feedforward,
                dropout=self.cfg.model.dropout,
                activate=self.cfg.model.activate,
                timestep_type=self.cfg.model.timestep_type,
                context_dim=text_cond_dim,  # Text instead of floor plan condition
                mlp_type=self.cfg.model.mlp_type,
            )

            if self.cfg.ema.use:
                self.ema = EMAModel(
                    model=self.model,
                    update_after_step=self.cfg.ema.update_after_step,
                    inv_gamma=self.cfg.ema.inv_gamma,
                    power=self.cfg.ema.power,
                    min_value=self.cfg.ema.min_value,
                    max_value=self.cfg.ema.max_value,
                )

        def predict_noise(
            self,
            noisy_scenes: torch.Tensor,
            timesteps: Union[torch.IntTensor, int],
            cond_dict: Dict[str, Any] = None,
            use_ema: bool = False,
        ) -> torch.Tensor:
            """Predict the noise for a batch of noisy scenes.

            Args:
                noisy_scenes (torch.Tensor): Input of shape (B, N, V) where N are the
                    number of objects and V is the object feature vector length.
                timesteps (Union[torch.IntTensor, int]): The diffusion step to condition
                    the denoising on.
                cond_dict: Dict[str, Any]: The dict containing the conditioning
                    information.
                use_ema (bool): Whether to use the EMA model.

            Returns:
                torch.Tensor: Output of same shape as the input.
            """
            assert not (use_ema and not self.cfg.ema.use)
            model = self.ema.model if use_ema else self.model

            # Process different timestep input formats.
            if not torch.is_tensor(timesteps):
                # Preferably, timesteps should be a tensor to avoid device issues.
                timesteps = torch.tensor(
                    [timesteps], dtype=torch.long, device=self.device
                )
                # Broadcast to batch dimension.
                timesteps = timesteps.expand(noisy_scenes.size(0))  # Shape (B,)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(self.device)
                # Broadcast to batch dimension.
                timesteps = timesteps.expand(noisy_scenes.size(0))  # Shape (B,)

            text_cond = None
            if cond_dict is not None and self.txt_encoder is not None:
                # Use bfloat16 for text encoder.
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=(
                        torch.bfloat16 if self.device.type == "cuda" else torch.float32
                    ),
                ):
                    text_cond: torch.Tensor = self.txt_encoder(
                        cond_dict["text_cond"]
                    )  # Shape (B, max_length, C)
                    # Average over the sequence length.
                    text_cond = text_cond.mean(dim=1)  # Shape (B, C)
                    # Convert to lightning dtype.
                    text_cond = text_cond.to(noisy_scenes.dtype)

                    # Expand context along num_objects dimension.
                    text_cond = text_cond.unsqueeze(1).expand(
                        -1, noisy_scenes.size(1), -1
                    )  # Shape (B, N, C)

            # Predict the noise.
            predicted_noise = model(
                noisy_scenes, time=timesteps, context=text_cond, context_cross=None
            )  # Shape (B, N, V)

            return predicted_noise

        def put_model_in_eval_mode(self) -> None:
            """Put the denoising model in evaluation mode."""
            self.model.eval()
            if self.cfg.ema.use:
                self.ema.eval()

        def on_train_batch_end(self, outputs, batch, batch_idx):
            if self.cfg.ema.use:
                self.ema.step(self.model)

    return SceneDiffuserMiDiffusion
