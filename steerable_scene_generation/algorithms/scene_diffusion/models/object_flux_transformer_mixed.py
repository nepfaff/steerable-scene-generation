from typing import Optional, Tuple, Union

import einops
import torch

from torch import nn

from .object_flux_transformer import MLPEmbedder, ObjectFluxTransformer


class ObjectFluxTransformerMixed(ObjectFluxTransformer):
    """
    Object transformer inspired by Flux from Black Forest.

    Uses double-stream transformer layers for cross-modal attention between scene and
    text embeddings if text_cond_dim is not None.

    A transformer for processing sequences of objects. The input is assumed to be an
    un-ordered set of objects. The input has shape (B, num_objects, feature_dim) and the
    output has the same shape.
    """

    def __init__(
        self,
        continous_input_dim: int,
        max_num_objects: int,
        embedding_dim: int,
        hidden_dim: int,
        mlp_ratio: float,
        num_heads: int,
        num_single_layers: int,
        num_double_layers: int,
        concatenate_input_features: bool,
        head_dim: Optional[int] = None,
        cond_dim: Optional[int] = None,
        text_cond_dim: Optional[int] = None,
    ):
        """
        Args:
            continous_input_dim (int): The number of continuous input channels. This
                corresponds to the continous feature dimension of the input objects.
            max_num_objects (int): The maximum number of objects per scene. The number
                of discrete classes is this number + 2 as it also includes the [empty]
                object and [mask] token.
            embedding_dim (int): The dimension that both the continous and discrete
                inputs are projected to before they are being passed to the network.
                This must be divisible by the number of groups for group normalization.
            hidden_dim (int): Dimension of the hidden layers in the transformer.
            mlp_ratio (float): Ratio of the MLP hidden dimension to the hidden size.
            num_heads (int): Number of attention heads in each transformer layer.
            num_single_layers (int): Number of single-stream transformer layers.
            num_double_layers (int): Number of double-stream transformer layers. Ignored
                if text_cond_dim is None.
            concatenate_input_features (bool, optional): Whether to concatenate
                continous, discrete features instead of adding them. In this case, the
                embedding dimension before concatenation is `embedding_dim` / 2.
            head_dim (Optional[int]): Dimension of each attention head. If None, it
                defaults to hidden_dim // num_heads.
            cond_dim (Optional[int]): Dimension of the conditioning vector.
            text_cond_dim (Optional[int]): Dimension of the text conditioning vector.
        """
        if concatenate_input_features:
            assert embedding_dim % 2 == 0

        super().__init__(
            object_feature_dim=embedding_dim,
            hidden_dim=hidden_dim,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            num_single_layers=num_single_layers,
            num_double_layers=num_double_layers,
            head_dim=head_dim,
            cond_dim=cond_dim,
            text_cond_dim=text_cond_dim,
        )

        self.max_num_objects = max_num_objects
        self.concatenate_input_features = concatenate_input_features

        init_embedding_dim = (
            embedding_dim // 2 if concatenate_input_features else embedding_dim
        )

        # Input projection for the continous component.

        self.continous_embedding = MLPEmbedder(
            in_dim=continous_input_dim, hidden_dim=init_embedding_dim
        )

        # Input projection for the discrete component.
        # Add an extra embeddings for the [empty] object and D3PM [mask] token.
        self.object_model_emb = nn.Embedding(max_num_objects + 2, init_embedding_dim)

        # Output projection for the continous component.
        self.to_continous_out = MLPEmbedder(
            in_dim=embedding_dim, hidden_dim=continous_input_dim
        )

        # Output projection for the discrete component. Don't include a dimension for
        # the [mask] token as x0 can't contain any [mask] tokens.
        self.to_logits = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, max_num_objects + 1),
        )

    def forward(
        self,
        x_continous: torch.Tensor,
        x_discrete: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: Optional[torch.Tensor] = None,
        text_cond: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_continous (torch.Tensor): The continous input of shape (B, num_objects,
                feature_dim).
            x_discrete (torch.LongTensor): The discrete input of shape (B, num_objects)
                where values are non-zero integers that represent discrete classes.
            timestep (Union[torch.LongTensor, float, int]): The diffusion step to
                condition the denoising on.
            cond (torch.Tensor | None): The conditioning tensor of shape (B, cond_dim).
            text_cond (torch.Tensor | None): The text conditioning tensor of shape
                (B, sequence_length, text_cond_dim). This should come from a text
                encoder such as T5.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of
                - continuous_out: The continous output of shape (B, num_objects,
                    feature_dim).
                - discrete_out: The discrete output of shape (B,
                    num_discrete_diffusion_classes-1, N). This doesn't include the
                    class for the [mask] token.
        """
        # Input projection.
        x_continous_embedding = self.continous_embedding(
            x_continous
        )  # Shape (B, num_objects, init_emb_dim)
        x_discrete_embedding = self.object_model_emb(
            x_discrete
        )  # Shape (B, num_objects, init_emb_dim)

        # denoising_input has shape (B, num_objects, emb_dim).
        if self.concatenate_input_features:
            denoising_input = torch.concat(
                [x_continous_embedding, x_discrete_embedding], dim=1
            )
        else:
            denoising_input = x_continous_embedding + x_discrete_embedding

        # Denoising.
        denoising_output = super().forward(
            sample=denoising_input,
            timestep=timestep,
            cond=cond,
            text_cond=text_cond,
        )  # Shape (B, num_objects, emb_dim)

        # Continous output projection.
        x_continous_out = self.to_continous_out(
            denoising_output
        )  # Shape (B, num_objects, feature_dim)

        # Discrete output projection.
        x_discrete_out = self.to_logits(
            denoising_output
        )  # Shape (B, num_objects, max_num_objects+1)
        x_discrete_out = einops.rearrange(
            x_discrete_out, "b n c -> b c n"
        )  # Shape (B, max_num_objects+1, num_objects)

        return x_continous_out, x_discrete_out
