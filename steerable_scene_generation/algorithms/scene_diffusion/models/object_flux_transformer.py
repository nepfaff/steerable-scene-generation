"""
Part of this code has been adapted from https://github.com/black-forest-labs/flux.
"""

import logging
import math

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from einops import rearrange
from torch import nn

console_logger = logging.getLogger(__name__)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Computes scaled dot-product attention.

    Args:
        q (torch.Tensor): Query tensor of shape (B, H, L, D).
        k (torch.Tensor): Key tensor of shape (B, H, L, D).
        v (torch.Tensor): Value tensor of shape (B, H, L, D).

    Returns:
        torch.Tensor: Output tensor of shape (B, L, H * D).
    """
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")
    return x


def timestep_embedding(
    t: torch.Tensor, dim: int, max_period: int = 10000, time_factor: float = 1000.0
) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    Args:
        t (torch.Tensor): A 1-D torch.Tensor of N indices, one per batch element. These
            may be fractional.
        dim (int): The dimension of the output.
        max_period (int): Controls the minimum frequency of the embeddings.

    Returns:
        An (N, D) torch.Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(t.device)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    """
    A multi-layer perceptron (MLP) embedder with a hidden layer using SiLU activation.
    """

    def __init__(self, in_dim: int, hidden_dim: int):
        """
        Args:
            in_dim (int): Dimension of the input features.
            hidden_dim (int): Dimension of the hidden layer and output features.
        """
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, in_dim).

        Returns:
            torch.Tensor: Output tensor of shape (B, hidden_dim).
        """
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization without centering."""

    def __init__(self, dim: int):
        """
        Args:
            dim (int): Dimension of the input features to normalize.
        """
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (..., dim).

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        # Reshape self.scale for broadcasting if necessary.
        scale = self.scale.view(*([1] * (x.dim() - 1)), -1)
        return (x * rrms).to(dtype=x_dtype) * scale


class QKNorm(torch.nn.Module):
    """Module that applies RMS normalization to query and key tensors separately."""

    def __init__(self, dim: int):
        """
        Args:
            dim (int): Dimension of the query and key features.
        """
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q (torch.Tensor): Query tensor of shape (..., dim).
            k (torch.Tensor): Key tensor of shape (..., dim).
            v (torch.Tensor): Value tensor (used for device and dtype alignment).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Normalized query and key tensors.
        """
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


@dataclass
class ModulationOut:
    """Dataclass to store the output of the Modulation module."""

    shift: torch.Tensor
    """Shift tensor of shape (B, 1, dim)."""
    scale: torch.Tensor
    """scale (torch.Tensor): Scale tensor of shape (B, 1, dim)."""
    gate: torch.Tensor
    """gate (torch.Tensor): Gate tensor of shape (B, 1, dim)."""


class Modulation(nn.Module):
    """
    Modulation module that generates shift, scale, and gate tensors from a conditioning
    vector.
    """

    def __init__(self, dim: int, is_double: bool = False):
        """
        Args:
            dim (int): Dimension of the input vector and output tensors.
            is_double (bool): Whether to use a double-sized modulation.
        """
        super().__init__()
        self.is_double = is_double
        self.multiplier = 6 if is_double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(
        self, vec: torch.Tensor
    ) -> Tuple[ModulationOut, ModulationOut] | ModulationOut:
        """
        Args:
            vec (torch.Tensor): Input conditioning vector of shape (B, dim).

        Returns:
            ModulationOut: A dataclass containing shift, scale, and gate tensors of
            shape (B, 1, dim).
        """
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(
            self.multiplier, dim=-1
        )

        if self.is_double:
            return ModulationOut(*out[:3]), ModulationOut(*out[3:])
        return ModulationOut(*out[:3])


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        mlp_ratio: float = 4.0,
        qk_scale: Optional[float] = None,
    ):
        """
        Initializes the SingleStreamBlock.

        Args:
            hidden_dim (int): Dimension of the hidden size.
            num_heads (int): Number of attention heads.
            head_dim (Optional[int]): Dimension of each attention head. If None,
                defaults to hidden_dim // num_heads.
            mlp_ratio (float): Ratio of MLP hidden dimension to hidden_dim.
            qk_scale (Optional[float]): Scaling factor for q and k. If None, set to
                1 / sqrt(head_dim).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else hidden_dim // num_heads

        # Validate dimensions.
        if hidden_dim < self.num_heads * self.head_dim:
            raise ValueError(
                f"hidden_size ({hidden_dim}) must be >= num_heads ({num_heads}) * "
                f"head_dim ({self.head_dim})"
            )

        self.scale = qk_scale or self.head_dim**-0.5
        self.mlp_hidden_dim = int(hidden_dim * mlp_ratio)

        # Calculate the dimension for qkv (queries, keys, values).
        self.qkv_dim = self.num_heads * self.head_dim * 3  # For q, k, v

        # linear1 outputs qkv and mlp_in.
        self.linear1 = nn.Linear(hidden_dim, self.qkv_dim + self.mlp_hidden_dim)

        # Attention output dimension.
        attn_output_dim = self.num_heads * self.head_dim

        # linear2 combines attention output and MLP output.
        self.linear2 = nn.Linear(attn_output_dim + self.mlp_hidden_dim, hidden_dim)

        self.norm = QKNorm(self.head_dim)

        self.pre_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SingleStreamBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, hidden_size).
            cond (torch.Tensor): Conditioning tensor of shape (B, hidden_size).

        Returns:
            torch.Tensor: Output tensor of shape (B, L, hidden_size).
        """
        # Compute and apply modulation.
        mod = self.modulation(cond)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift

        # Apply the first linear layer and split into qkv and MLP parts.
        qkv, mlp = torch.split(
            self.linear1(x_mod), [self.qkv_dim, self.mlp_hidden_dim], dim=-1
        )

        # Reshape qkv for multi-head attention.
        qkv = rearrange(
            qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads, D=self.head_dim
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply normalization to q and k.
        q, k = self.norm(q, k, v)

        # Compute attention.
        attn = attention(q, k, v)  # Shape (B, L, num_heads * head_dim)

        # Apply MLP activation.
        mlp_out = self.mlp_act(mlp)

        # Concatenate attention and MLP outputs.
        combined = torch.cat(
            (attn, mlp_out), dim=-1
        )  # Shape (B, L, attn_output_dim + mlp_hidden_dim)

        # Apply the second linear layer.
        output = self.linear2(combined)

        # Apply modulation gate and residual connection.
        return x + mod.gate * output


class DoubleStreamBlock(nn.Module):
    """
    A block for processing two input streams (scene and text embeddings) with
    cross-modal attention.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        mlp_ratio: float = 4.0,
        qk_scale: Optional[float] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else hidden_dim // num_heads

        # Validate dimensions.
        if hidden_dim < self.num_heads * self.head_dim:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be >= num_heads ({num_heads}) * "
                f"head_dim ({self.head_dim})"
            )

        self.scale = qk_scale or self.head_dim**-0.5
        self.mlp_hidden_dim = int(hidden_dim * mlp_ratio)

        # Calculate the dimension for qkv (queries, keys, values).
        qkv_dim = self.num_heads * self.head_dim * 3  # For q, k, v

        # Scene-specific layers.
        self.x_modulation = Modulation(hidden_dim, is_double=True)
        self.x_pre_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.x_qkv_linear = nn.Linear(hidden_dim, qkv_dim)
        self.x_qk_norm = QKNorm(self.head_dim)
        self.x_post_att_linear = nn.Linear(num_heads * self.head_dim, hidden_dim)
        self.x_mlp = nn.Sequential(
            nn.Linear(hidden_dim, self.mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.mlp_hidden_dim, hidden_dim),
        )
        self.x_post_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)

        # Text-specific layers.
        self.txt_modulation = Modulation(hidden_dim, is_double=True)
        self.txt_pre_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.txt_qkv_linear = nn.Linear(hidden_dim, qkv_dim)
        self.txt_qk_norm = QKNorm(self.head_dim)
        self.txt_post_att_linear = nn.Linear(num_heads * self.head_dim, hidden_dim)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_dim, self.mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.mlp_hidden_dim, hidden_dim),
        )
        self.txt_post_norm = nn.LayerNorm(
            hidden_dim, elementwise_affine=False, eps=1e-6
        )

        # Modality embeddings.
        self.scene_modality_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.text_modality_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(
        self, x: torch.Tensor, txt: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the DoubleStreamBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L_x, hidden_dim).
            txt (torch.Tensor): Text tensor of shape (B, L_txt, hidden_dim).
            cond (torch.Tensor): Conditioning tensor of shape (B, hidden_size).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Processed x of shape (B, L_x, hidden_dim)
                and processed txt of shape (B, L_txt, hidden_dim).
        """
        # Add modality embeddings.
        x = x + self.scene_modality_embedding
        txt = txt + self.text_modality_embedding

        # Modulation and normalization for x.
        x_mod1, x_mod2 = self.x_modulation(cond)
        x_modulated = (1 + x_mod1.scale) * self.x_pre_norm(x) + x_mod1.shift
        x_qkv = self.x_qkv_linear(x_modulated)
        x_qkv = rearrange(
            x_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads, D=self.head_dim
        )
        x_q, x_k, x_v = x_qkv[0], x_qkv[1], x_qkv[2]
        x_q, x_k = self.x_qk_norm(x_q, x_k, x_v)

        # Modulation and normalization for text.
        txt_mod1, txt_mod2 = self.txt_modulation(cond)
        txt_modulated = (1 + txt_mod1.scale) * self.txt_pre_norm(txt) + txt_mod1.shift
        txt_qkv = self.txt_qkv_linear(txt_modulated)
        txt_qkv = rearrange(
            txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads, D=self.head_dim
        )
        txt_q, txt_k, txt_v = txt_qkv[0], txt_qkv[1], txt_qkv[2]
        txt_q, txt_k = self.txt_qk_norm(txt_q, txt_k, txt_v)

        # Cross-modal attention.
        q = torch.cat((txt_q, x_q), dim=2)
        k = torch.cat((txt_k, x_k), dim=2)
        v = torch.cat((txt_v, x_v), dim=2)
        attn = attention(q, k, v)  # Shape: (B, L_txt + L_x, num_heads * head_dim)

        # Split attention outputs back into text and scene streams.
        txt_attn, x_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # Post-attention processing for scene.
        x_out = x + x_mod1.gate * self.x_post_att_linear(x_attn)
        x_out = x_out + x_mod2.gate * self.x_mlp(
            (1 + x_mod2.scale) * self.x_post_norm(x_out) + x_mod2.shift
        )  # Shape (B, L_x, hidden_dim)

        # Post-attention processing for text.
        txt_out = txt + txt_mod1.gate * self.txt_post_att_linear(txt_attn)
        txt_out = txt_out + txt_mod2.gate * self.txt_mlp(
            (1 + txt_mod2.scale) * self.txt_post_norm(txt_out) + txt_mod2.shift
        )  # Shape (B, L_txt, hidden_dim)

        return x_out, txt_out


class LastLayer(nn.Module):
    def __init__(self, hidden_dim: int, out_channels: int):
        """
        Args:
            hidden_dim (int): Dimension of the hidden size.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, hidden_dim).
            vec (torch.Tensor): Conditioning vector of shape (B, hidden_dim).

        Returns:
            torch.Tensor: Output tensor of shape (B, L, out_channels).
        """
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


class ObjectFluxTransformer(nn.Module):
    """
    Object transformer inspired by Flux from Black Forest.

    Uses double-stream transformer layers for cross-modal attention between scene and
    text embeddings if text_cond_dim is not None.

    A transformer network for processing sequences of objects. The input is
    assumed to be an ordered or unordered set of objects. The input has shape
    (B, num_objects, feature_dim) and the output has the same shape. Permuation
    equivariance across the `num_object` dimension is preserved.
    """

    def __init__(
        self,
        object_feature_dim: int,
        hidden_dim: int,
        mlp_ratio: float,
        num_heads: int,
        num_single_layers: int,
        num_double_layers: int,
        head_dim: Optional[int] = None,
        cond_dim: Optional[int] = None,
        text_cond_dim: Optional[int] = None,
        object_feature_dim_out: Optional[int] = None,
    ):
        """
        Args:
            object_feature_dim (int): Dimension of each object's feature in the input.
            hidden_dim (int): Dimension of the hidden layers in the transformer.
            mlp_ratio (float): Ratio of the MLP hidden dimension to the hidden size.
            num_heads (int): Number of attention heads in each transformer layer.
            num_single_layers (int): Number of single-stream transformer layers.
            num_double_layers (int): Number of double-stream transformer layers. Ignored
                if text_cond_dim is None.
            head_dim (Optional[int]): Dimension of each attention head. If None, it
                defaults to hidden_dim // num_heads.
            cond_dim (Optional[int]): Dimension of the conditioning vector.
            text_cond_dim (Optional[int]): Dimension of the text conditioning vector.
            object_feature_dim_out (Optional[int]): Dimension of the object
                feature output. If None, then it is the same as the input
                object_feature_dim.
        """
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_dim} must be divisible by num_heads {num_heads}"
            )
        if num_double_layers > 0 and text_cond_dim is None:
            console_logger.warning(
                "num_double_layers is ignored because text_cond_dim is None."
            )

        self.scene_in = nn.Linear(object_feature_dim, hidden_dim)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=hidden_dim)
        if cond_dim is not None:
            self.vector_in = MLPEmbedder(in_dim=cond_dim, hidden_dim=hidden_dim)
        if text_cond_dim is not None:
            self.text_in = nn.Linear(text_cond_dim, hidden_dim)

        self.cond_dim = cond_dim
        self.text_cond_dim = text_cond_dim

        self.double_blocks = (
            nn.ModuleList(
                [
                    DoubleStreamBlock(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        mlp_ratio=mlp_ratio,
                    )
                    for _ in range(num_double_layers)
                ]
            )
            if text_cond_dim is not None
            else None
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(num_single_layers)
            ]
        )

        if object_feature_dim_out is None:
            object_feature_dim_out = object_feature_dim
        self.final_layer = LastLayer(hidden_dim, object_feature_dim_out)

    def get_timestep_feature(
        self,
        timestep: Union[torch.Tensor, float, int],
        batch_dim: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Processes and encodes the timestep(s). Returns the timestep encoding of shape
        (B, dsed)."""
        # Process different timestep input formats.
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # Preferably, timesteps should be a tensor to avoid device issues.
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(device)

        # Broadcast to batch dimension.
        timesteps = timesteps.expand(batch_dim)  # Shape (B,)

        # Encode the diffusion step for conditioning.
        global_timestep_feature = self.time_in(
            timestep_embedding(timesteps, 256)
        )  # Shape (B, dsim)

        return global_timestep_feature

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: torch.Tensor | None = None,
        text_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            sample (torch.Tensor): The input of shape (B, num_objects, feature_dim).
            timestep (Union[torch.Tensor, float, int]): The diffusion step to condition
                the denoising on.
            cond (torch.Tensor | None): The conditioning tensor of shape (B, cond_dim).
            text_cond (torch.Tensor | None): The text conditioning tensor of shape
                (B, sequence_length, text_cond_dim). This should come from a text
                encoder such as T5.

        Returns:
            torch.Tensor: The output of same shape as the input.
        """
        sample = self.scene_in(sample)

        # Timestep conditioning.
        global_timestep_feature = self.get_timestep_feature(
            timestep=timestep, batch_dim=sample.shape[0], device=sample.device
        )  # Shape (B, dsed)

        # Combine input conditioning and timestep conditioning.
        cond_vec = global_timestep_feature
        if cond is not None:
            cond_vec += self.vector_in(cond)

        if text_cond is not None:
            txt_vec = self.text_in(text_cond)

            for block in self.double_blocks:
                sample, txt_vec = block(sample, txt_vec, cond_vec)

            sample = torch.cat(
                (txt_vec, sample), dim=1
            )  # Shape (B, txt_seq_len + num_objects, hidden_dim)

        for block in self.single_blocks:
            sample = block(sample, cond=cond_vec)

        if text_cond is not None:
            sample = sample[
                :, txt_vec.shape[1] :, ...
            ]  # Shape (B, num_objects, hidden_dim)

        sample = self.final_layer(
            sample, cond_vec
        )  # Shape (B, num_objects, feature_dim)
        return sample
