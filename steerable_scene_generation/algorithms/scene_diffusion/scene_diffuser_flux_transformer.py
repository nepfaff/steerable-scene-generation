from typing import Any, Dict, Type, Union

import torch

from steerable_scene_generation.algorithms.common.ema_model import EMAModel
from steerable_scene_generation.algorithms.common.txt_encoding import (
    load_txt_encoder_from_config,
)
from steerable_scene_generation.datasets.scene.scene import SceneDataset

from .models import ObjectFluxTransformer
from .scene_diffuser_base_continous import SceneDiffuserBaseContinous


def create_scene_diffuser_flux_transformer(
    trainer_class: Type[SceneDiffuserBaseContinous],
) -> Type[SceneDiffuserBaseContinous]:
    """
    Factory function to create a scene diffuser flux transformer model class.

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

    class SceneDiffuserFluxTransformer(trainer_class):
        """
        Scene diffusion on a set of un-ordered objects. The number of objects and types
        of objects are not fixed. The object vectors consist of [translation, rotation,
        model_vector]. All scenes have `max_num_objects_per_scene` objects.
        """

        def __init__(self, cfg, dataset: SceneDataset):
            """
            cfg is a DictConfig object defined by
            `configurations/algorithm/scene_diffuser_flux_transformer.yaml`.
            """
            super().__init__(cfg, dataset=dataset)

        def _build_model(self):
            """Create all pytorch models."""
            super()._build_model()

            obj_vec_len = self.scene_vec_desc.get_object_vec_len()
            obj_diff_vec_len = self.scene_vec_desc.get_diff_vec_len()

            # Conditioning.
            self.txt_encoder, self.txt_encoder_coarse = None, None
            if self.cfg.classifier_free_guidance.use:
                self.txt_encoder, text_cond_dim = load_txt_encoder_from_config(
                    self.cfg, component="encoder"
                )
            else:
                text_cond_dim = None
            use_coarse = (
                self.cfg.classifier_free_guidance.txt_encoder_coarse is not None
            )
            if use_coarse:
                (
                    self.txt_encoder_coarse,
                    text_cond_dim_coarse,
                ) = load_txt_encoder_from_config(
                    self.cfg, is_coarse=True, component="encoder"
                )
            else:
                text_cond_dim_coarse = None

            self.model = ObjectFluxTransformer(
                object_feature_dim=obj_vec_len,
                hidden_dim=self.cfg.model.hidden_dim,
                mlp_ratio=self.cfg.model.mlp_ratio,
                num_single_layers=self.cfg.model.num_single_layers,
                num_double_layers=self.cfg.model.num_double_layers,
                num_heads=self.cfg.model.num_heads,
                head_dim=self.cfg.model.head_dim,
                cond_dim=text_cond_dim_coarse,
                text_cond_dim=text_cond_dim,
                object_feature_dim_out=obj_diff_vec_len,
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

            text_cond, text_cond_coarse = None, None
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
                    # Convert to lightning dtype.
                    text_cond = text_cond.to(noisy_scenes.dtype)

                    if self.txt_encoder_coarse is not None:
                        text_cond_coarse: torch.Tensor = self.txt_encoder_coarse(
                            cond_dict["text_cond_coarse"]
                        )  # Shape (B, max_length, C_coarse)
                        text_cond_coarse = text_cond_coarse.to(noisy_scenes.dtype)
                        text_cond_coarse = text_cond_coarse.mean(
                            dim=1
                        )  # Shape (B, C_coarse)

            # Predict the noise.
            predicted_noise = model(
                noisy_scenes,
                timestep=timesteps,
                cond=text_cond_coarse,
                text_cond=text_cond,
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

    return SceneDiffuserFluxTransformer
