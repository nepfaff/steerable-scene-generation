from typing import Any, Dict, Type, Union

import torch

from steerable_scene_generation.algorithms.common.ema_model import EMAModel
from steerable_scene_generation.algorithms.common.txt_encoding import (
    load_txt_encoder_from_config,
)
from steerable_scene_generation.datasets.scene.scene import SceneDataset

from .models import ObjectFluxTransformerMixed
from .scene_diffuser_base_mixed import SceneDiffuserBaseMixed


def create_scene_diffuser_mixed_flux_transformer(
    trainer_class: Type[SceneDiffuserBaseMixed],
) -> Type[SceneDiffuserBaseMixed]:
    """
    Factory function to create a mixed continous, discrete diffusion scene diffuser
    Flux-style transformer class.

    Args:
        trainer_class (Type[SceneDiffuserBaseMixed]): The base class for the scene
            diffuser. This class should be a subclass of SceneDiffuserBaseMixed and
            implement the forward method and any additional methods needed for the
            training method.

    Returns:
        SceneDiffuser: The scene diffuser class. This class implements any
            model-specific logic needed for the scene diffusion algorithm. See the
            docstring of the returned class for more details.
    """

    class SceneDiffuserMixedFluxTransformer(trainer_class):
        """
        Mixed scene diffusion on a set of un-ordered objects. The number of objects and
        types of objects are not fixed. The object vectors consist of [translation,
        rotation, model_vector]. All scenes have `max_num_objects_per_scene` objects.
        """

        def __init__(self, cfg, dataset: SceneDataset):
            """
            cfg is a DictConfig object defined by
            `configurations/algorithm/scene_diffuser_mixed_flux_transformer.yaml`.
            """
            super().__init__(cfg, dataset=dataset)

        def _build_model(self):
            """Create all pytorch models."""
            super()._build_model()

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

            obj_vec_len = self.scene_vec_desc.get_object_vec_len()
            discrete_vec_len = self.scene_vec_desc.model_path_vec_len
            continous_vec_len = obj_vec_len - discrete_vec_len
            self.model = ObjectFluxTransformerMixed(
                continous_input_dim=continous_vec_len,
                max_num_objects=discrete_vec_len - 1,  # One-hot vector without [empty]
                embedding_dim=self.cfg.model.embedding_dim,
                hidden_dim=self.cfg.model.hidden_dim,
                mlp_ratio=self.cfg.model.mlp_ratio,
                num_single_layers=self.cfg.model.num_single_layers,
                num_double_layers=self.cfg.model.num_double_layers,
                concatenate_input_features=self.cfg.model.concatenate_input_features,
                num_heads=self.cfg.model.num_heads,
                head_dim=self.cfg.model.head_dim,
                cond_dim=text_cond_dim_coarse,
                text_cond_dim=text_cond_dim,
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

        def denoise(
            self,
            x_continous: torch.Tensor,
            x_discrete: torch.Tensor,
            timesteps: Union[torch.IntTensor, int],
            cond_dict: Dict[str, Any] = None,
            use_ema: bool = False,
        ) -> Union[torch.Tensor, torch.Tensor]:
            """For a batch of noisy scenes, predict the noise for the continous part and
            x0 for the discrete part.

            Args:
                x_continous (torch.Tensor): Continuous scenes of shape (B, N, Vc) where
                    N are the number of objects and Vc is the continous object feature
                    vector length.
                x_discrete (torch.Tensor): The discrete input of shape (B, N) where
                    values are non-zero integers that represent discrete classes.
                timesteps (Union[torch.IntTensor, int]): The diffusion step to condition
                    the denoising on.
                cond_dict: Dict[str, Any]: The dict containing the conditioning
                    information.
                use_ema (bool): Whether to use the EMA model.

            Returns:
                Union[torch.Tensor, torch.Tensor]: A tuple of
                    - continous_output: Noise prediction for `x_continous` of of shape
                        (B, N, Vc).
                    - discrete_output: x0 prediction for `x_discrete` of shape (B,
                        num_discrete_diffusion_classes-1, N). This doesn't include the
                        class for the [mask] token.
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
                    text_cond = text_cond.to(x_continous.dtype)

                    if self.txt_encoder_coarse is not None:
                        text_cond_coarse: torch.Tensor = self.txt_encoder_coarse(
                            cond_dict["text_cond_coarse"]
                        )  # Shape (B, max_length, C_coarse)
                        text_cond_coarse = text_cond_coarse.to(x_continous.dtype)
                        text_cond_coarse = text_cond_coarse.mean(
                            dim=1
                        )  # Shape (B, C_coarse)

            # Predict the noise.
            x_continous_out, x_discrete_out = model(
                x_continous=x_continous,
                x_discrete=x_discrete,
                timestep=timesteps,
                cond=text_cond_coarse,
                text_cond=text_cond,
            )  # First has shape (B, N, Vc) and second has shape (B, N, Vd)

            return x_continous_out, x_discrete_out

        def put_model_in_eval_mode(self) -> None:
            """Put the denoising model in evaluation mode."""
            self.model.eval()
            if self.cfg.ema.use:
                self.ema.eval()

        def on_train_batch_end(self, outputs, batch, batch_idx):
            if self.cfg.ema.use:
                self.ema.step(self.model)

    return SceneDiffuserMixedFluxTransformer
