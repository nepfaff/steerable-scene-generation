"""
The code in this file has been copied from MiDiffusion:
https://github.com/MIT-SPARK/MiDiffusion/blob/main/midiffusion/networks/denoising_net/continuous_transformer.py
and the related files.

We didn't make any changes to the code apart from applying our formatters for the code
to pass the CI checks. We avoided cleaning up the code or making other changes to prevent
us from accidentally deviating from the original model.
"""

import math

import torch
import torch.nn.functional as F

from einops.layers.torch import Rearrange
from torch import Tensor, nn

LAYER_NROM_EPS = 1e-5  # pytorch's default: 1e-5


class SinusoidalPosEmb(nn.Module):
    """https://github.com/microsoft/VQ-Diffusion/blob/main/image_synthesis/modeling/transformers/transformer_utils.py"""

    def __init__(self, dim: int, num_steps: int = 4000, rescale_steps: int = 4000):
        super().__init__()
        self.dim = dim
        if num_steps != rescale_steps:
            self.num_steps = float(num_steps)
            self.rescale_steps = float(rescale_steps)
            self.input_scaling = True
        else:
            self.input_scaling = False

    def forward(self, x: Tensor):
        # (B) -> (B, self.dim)
        if self.input_scaling:
            x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class GELU2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * F.sigmoid(1.702 * x)


class _AdaNorm(nn.Module):
    """Base normalization layer that incorporate timestep embeddings"""

    def __init__(
        self, n_embd: int, max_timestep: int, emb_type: str = "adalayernorm_abs"
    ):
        super().__init__()
        assert n_embd % 2 == 0
        if "abs" in emb_type:
            self.emb = SinusoidalPosEmb(n_embd, num_steps=max_timestep)
        elif "mlp" in emb_type:
            self.emb = nn.Sequential(
                Rearrange("b -> b 1"),
                nn.Linear(1, n_embd // 2),
                nn.ReLU(),
                nn.Linear(n_embd // 2, n_embd),
            )
        else:
            self.emb = nn.Embedding(max_timestep, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)


class AdaLayerNorm(_AdaNorm):
    """Norm layer modified to incorporate timestep embeddings"""

    def __init__(
        self, n_embd: int, max_timestep: int, emb_type: str = "adalayernorm_abs"
    ):
        super().__init__(n_embd, max_timestep, emb_type)
        self.layernorm = nn.LayerNorm(
            n_embd, eps=LAYER_NROM_EPS, elementwise_affine=False
        )

    def forward(self, x: Tensor, timestep: Tensor):
        # (B, N, n_embd),(B,) -> (B, N, n_embd)
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)  # B, 1, 2*n_embd
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x


class AdaInsNorm(_AdaNorm):
    """Base instance normalization layer that incorporate timestep embeddings"""

    def __init__(
        self, n_embd: int, max_timestep: int, emb_type: str = "adainsnorm_abs"
    ):
        super().__init__(n_embd, max_timestep, emb_type)
        self.instancenorm = nn.InstanceNorm1d(n_embd, eps=LAYER_NROM_EPS)

    def forward(self, x: Tensor, timestep: Tensor):
        # (B, N, n_embd),(B,) -> (B, N, n_embd)
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)  # B, 1, 2*n_embd
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = (
            self.instancenorm(x.transpose(-1, -2)).transpose(-1, -2) * (1 + scale)
            + shift
        )
        return x


class SelfAttention(nn.Module):
    """Multi-head self attention block - in transformer encoder"""

    def __init__(self, n_embd, n_head, dropout=0.1, batch_first=True):
        super().__init__()
        assert n_embd % n_head == 0
        self.mha = nn.MultiheadAttention(
            n_embd, n_head, dropout, batch_first=batch_first
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        return self.mha(
            x,
            x,
            x,
            need_weights=False,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[0]


class CrossAttention(nn.Module):
    """Multi-head cross attention block - in transformer decoder"""

    def __init__(self, n_embd, n_head, dropout=0.1, batch_first=True, kv_embd=None):
        super().__init__()
        assert n_embd % n_head == 0
        self.mha = nn.MultiheadAttention(
            n_embd, n_head, dropout, batch_first=batch_first, kdim=kv_embd, vdim=kv_embd
        )

    def forward(self, q, kv, attn_mask=None, key_padding_mask=None):
        return self.mha(
            query=q,
            key=kv,
            value=kv,
            need_weights=False,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[0]


class Block(nn.Module):
    """Time-conditioned transformer block"""

    def __init__(
        self,
        n_embd=512,
        n_head=8,
        dim_feedforward=2048,
        dropout=0.1,
        activate="GELU",
        num_timesteps=1000,
        timestep_type="adalayernorm_abs",
        attn_type="self",
        num_labels=None,  # attn_type = 'selfcondition'
        label_type="adalayernorm",  # attn_type = 'selfcondition'
        cond_emb_dim=None,  # attn_type = 'selfcross'
        mlp_type="fc",
    ):
        super().__init__()
        self.attn_type = attn_type

        if "adalayernorm" in timestep_type:
            self.ln1 = AdaLayerNorm(n_embd, num_timesteps, timestep_type)
        elif "adainnorm" in timestep_type:
            self.ln1 = AdaInsNorm(n_embd, num_timesteps, timestep_type)
        else:
            raise ValueError(f"timestep_type={timestep_type} not valid.")

        if attn_type == "self":
            self.attn = SelfAttention(n_embd=n_embd, n_head=n_head, dropout=dropout)
            self.ln2 = nn.LayerNorm(n_embd)
        elif attn_type == "selfcondition":  # conditioned on int labels
            self.attn = SelfAttention(n_embd=n_embd, n_head=n_head, dropout=dropout)
            if "adalayernorm" in label_type:
                self.ln2 = AdaLayerNorm(n_embd, num_labels, label_type)
            else:
                self.ln2 = AdaInsNorm(n_embd, num_labels, label_type)
        elif attn_type == "selfcross":  # cross attention with cond_emb
            self.attn1 = SelfAttention(n_embd=n_embd, n_head=n_head, dropout=dropout)
            self.attn2 = CrossAttention(
                n_embd=n_embd,
                n_head=n_head,
                dropout=dropout,
                kv_embd=cond_emb_dim,
            )
            if "adalayernorm" in timestep_type:
                self.ln1_1 = AdaLayerNorm(n_embd, num_timesteps, timestep_type)
            else:
                raise ValueError(f"timestep_type={timestep_type} not valid.")
            self.ln2 = nn.LayerNorm(n_embd)
        else:
            raise ValueError(f"attn_type={attn_type} not valid.")

        assert activate in ["GELU", "GELU2"]
        act = nn.GELU() if activate == "GELU" else GELU2()
        if mlp_type == "fc":
            self.mlp = nn.Sequential(
                nn.Linear(n_embd, dim_feedforward),
                act,
                nn.Linear(dim_feedforward, n_embd),
                nn.Dropout(dropout),
            )
        else:
            raise NotImplemented

    def forward(self, x, timestep, cond_output=None, mask=None):
        if self.attn_type == "self":
            x = x + self.attn(self.ln1(x, timestep), attn_mask=mask)
            x = x + self.mlp(self.ln2(x))
        elif self.attn_type == "selfcondition":
            x = x + self.attn(self.ln1(x, timestep), attn_mask=mask)
            x = x + self.mlp(self.ln2(x, cond_output))
        elif self.attn_type == "selfcross":
            x = x + self.attn1(self.ln1(x, timestep), attn_mask=mask)
            x = x + self.attn2(self.ln1_1(x, timestep), cond_output, attn_mask=mask)
            x = x + self.mlp(self.ln2(x))
        else:
            return NotImplemented
        return x


class DenoiseTransformer(nn.Module):
    """Base denoising transformer class"""

    def __init__(
        self,
        n_layer=4,
        n_embd=512,
        n_head=8,
        dim_feedforward=2048,
        dropout=0.1,
        activate="GELU",
        num_timesteps=1000,
        timestep_type="adalayernorm_abs",
        context_dim=256,
        mlp_type="fc",
    ):
        super().__init__()

        # transformer backbone
        if context_dim == 0:
            self.tf_blocks = nn.Sequential(
                *[
                    Block(
                        n_embd,
                        n_head,
                        dim_feedforward,
                        dropout,
                        activate,
                        num_timesteps,
                        timestep_type,
                        mlp_type=mlp_type,
                        attn_type="self",
                    )
                    for _ in range(n_layer)
                ]
            )
        else:
            self.tf_blocks = nn.Sequential(
                *[
                    Block(
                        n_embd,
                        n_head,
                        dim_feedforward,
                        dropout,
                        activate,
                        num_timesteps,
                        timestep_type,
                        mlp_type=mlp_type,
                        attn_type="selfcross",
                        cond_emb_dim=context_dim,
                    )
                    for _ in range(n_layer)
                ]
            )

    @staticmethod
    def _encoder_mlp(hidden_size, input_size):
        mlp_layers = [
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        ]
        return nn.Sequential(*mlp_layers)

    @staticmethod
    def _decoder_mlp(hidden_size, output_size):
        mlp_layers = [
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
        ]
        return nn.Sequential(*mlp_layers)


class ContinuousDenoiseTransformer(DenoiseTransformer):
    """Continuous denoising transformer network where all object properties are
    treated as continuous"""

    def __init__(
        self,
        network_dim,
        seperate_all=True,
        n_layer=4,
        n_embd=512,
        n_head=8,
        dim_feedforward=2048,
        dropout=0.1,
        activate="GELU",
        num_timesteps=1000,
        timestep_type="adalayernorm_abs",
        context_dim=256,
        mlp_type="fc",
    ):
        # initialize self.tf_blocks, the transformer backbone
        super().__init__(
            n_layer,
            n_embd,
            n_head,
            dim_feedforward,
            dropout,
            activate,
            num_timesteps,
            timestep_type,
            context_dim,
            mlp_type,
        )

        # feature dimensions
        self.objectness_dim, self.class_dim, self.objfeat_dim = (
            network_dim["objectness_dim"],
            network_dim["class_dim"],
            network_dim["objfeat_dim"],
        )
        self.translation_dim, self.size_dim, self.angle_dim = (
            network_dim["translation_dim"],
            network_dim["size_dim"],
            network_dim["angle_dim"],
        )
        self.bbox_dim = self.translation_dim + self.size_dim + self.angle_dim
        self.channels = (
            self.bbox_dim + self.objectness_dim + self.class_dim + self.objfeat_dim
        )

        # Initial feature specific processing
        self.seperate_all = seperate_all
        if self.seperate_all:
            self.bbox_embedf = self._encoder_mlp(n_embd, self.bbox_dim)
            self.bbox_hidden2output = self._decoder_mlp(n_embd, self.bbox_dim)
            feature_str = "translation/size/angle"

            if self.class_dim > 0:
                self.class_embedf = self._encoder_mlp(n_embd, self.class_dim)
                feature_str += "/class"
            if self.objectness_dim > 0:
                self.objectness_embedf = self._encoder_mlp(n_embd, self.objectness_dim)
                feature_str += "/objectness"
            if self.objfeat_dim > 0:
                self.objfeat_embedf = self._encoder_mlp(n_embd, self.objfeat_dim)
                feature_str += "/objfeat"
            print("separate unet1d encoder/decoder of {}".format(feature_str))
        else:
            self.init_mlp = self._encoder_mlp(n_embd, self.channels)
            print("unet1d encoder of all object properties")

        # Final feature specific processing
        if self.seperate_all:
            self.bbox_hidden2output = self._decoder_mlp(n_embd, self.bbox_dim)
            if self.class_dim > 0:
                self.class_hidden2output = self._decoder_mlp(n_embd, self.class_dim)
            if self.objectness_dim > 0:
                self.objectness_hidden2output = self._decoder_mlp(
                    n_embd, self.objectness_dim
                )
            if self.objfeat_dim > 0:
                self.objfeat_hidden2output = self._decoder_mlp(n_embd, self.objfeat_dim)
        else:
            self.hidden2output = self._decoder_mlp(n_embd, self.channels)

    def forward(self, x, time, context=None, context_cross=None):
        # x: (B, N, C)
        if context_cross is not None:
            raise NotImplemented  # TODO

        # initial processing
        if self.seperate_all:
            x_bbox = self.bbox_embedf(x[:, :, 0 : self.bbox_dim])

            if self.class_dim > 0:
                start_index = self.bbox_dim
                x_class = self.class_embedf(
                    x[:, :, start_index : start_index + self.class_dim]
                )
            else:
                x_class = 0

            if self.objectness_dim > 0:
                start_index = self.bbox_dim + self.class_dim
                x_object = self.objectness_embedf(
                    x[:, :, start_index : start_index + self.objectness_dim]
                )
            else:
                x_object = 0

            if self.objfeat_dim > 0:
                start_index = self.bbox_dim + self.class_dim + self.objectness_dim
                x_objfeat = self.objfeat_embedf(
                    x[:, :, start_index : start_index + self.objfeat_dim]
                )
            else:
                x_objfeat = 0

            x = x_bbox + x_class + x_object + x_objfeat
        else:
            x = self.init_mlp(x)

        # transformer
        for block_idx in range(len(self.tf_blocks)):
            x = self.tf_blocks[block_idx](x, time, context)

        # final processing
        if self.seperate_all:
            out = self.bbox_hidden2output(x)
            if self.class_dim > 0:
                out_class = self.class_hidden2output(x)
                out = torch.cat([out, out_class], dim=2).contiguous()
            if self.objectness_dim > 0:
                out_object = self.objectness_hidden2output(x)
                out = torch.cat([out, out_object], dim=2).contiguous()
            if self.objfeat_dim > 0:
                out_objfeat = self.objfeat_hidden2output(x)
                out = torch.cat([out, out_objfeat], dim=2).contiguous()
        else:
            out = self.hidden2output(x)

        return out
