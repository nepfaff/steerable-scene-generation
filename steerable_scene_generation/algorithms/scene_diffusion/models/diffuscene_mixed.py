"""
The code in this file has been copied from DiffuScene:
https://github.com/tangjiapeng/DiffuScene/blob/master/scene_synthesis/networks/denoise_net.py
and modified to work with mixed discrete, continuous diffusion.

We tried to make the minimal changes to the original code to support mixed diffusion.
We used the existing encoding/ decoding MLPs for the added input and output projections.
We prefered code duplication with `diffuscene.py` to avoid making any changes to the
continuous implementation in `diffuscene.py` that corresponds to the DiffuScene paper.
"""

import torch

from torch import nn

from .diffuscene import Unet1D


class Unet1DMixed(Unet1D):
    def __init__(
        self,
        max_num_objects: int,
        embedding_dim: int,
        concatenate_input_features: bool = False,
        dim=256,  #
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        merge_bbox=False,
        objectness_dim=1,
        class_dim=21,
        translation_dim=3,
        size_dim=3,
        angle_dim=1,
        objfeat_dim=0,
        context_dim=256,
        instanclass_dim=0,
        modulate_time_context_instanclass=False,
        text_condition=False,
        text_dim=256,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
    ):
        super().__init__(
            dim=dim,
            init_dim=init_dim,
            out_dim=out_dim,
            dim_mults=dim_mults,
            channels=embedding_dim,  # Changed this
            self_condition=self_condition,
            seperate_all=False,  # Doesn't make sense for mixed model
            merge_bbox=merge_bbox,
            objectness_dim=objectness_dim,
            class_dim=class_dim,
            translation_dim=translation_dim,
            size_dim=size_dim,
            angle_dim=angle_dim,
            objfeat_dim=objfeat_dim,
            context_dim=context_dim,
            instanclass_dim=instanclass_dim,
            modulate_time_context_instanclass=modulate_time_context_instanclass,
            text_condition=text_condition,
            text_dim=text_dim,
            resnet_block_groups=resnet_block_groups,
            learned_variance=learned_variance,
            learned_sinusoidal_cond=learned_sinusoidal_cond,
            random_fourier_features=random_fourier_features,
            learned_sinusoidal_dim=learned_sinusoidal_dim,
        )

        self.max_num_objects = max_num_objects
        self.concatenate_input_features = concatenate_input_features

        init_embedding_dim = (
            embedding_dim // 2 if concatenate_input_features else embedding_dim
        )

        # Input projection for the continous component.
        continous_input_dim = translation_dim + size_dim + angle_dim
        self.continous_embedding_conv = self._encoder_mlp(
            init_embedding_dim, continous_input_dim
        )

        # Input projection for the discrete component.
        # Add an extra embeddings for the [empty] object and D3PM [mask] token.
        self.object_model_emb = nn.Embedding(max_num_objects + 2, init_embedding_dim)

        # Output projection for the continous component.
        self.to_continous_out = self._decoder_mlp(embedding_dim, continous_input_dim)

        # Output projection for the discrete component. Don't include a dimension for
        # the [mask] token as x0 can't contain any [mask] tokens.
        self.to_logits = nn.Sequential(
            nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, max_num_objects + 1)
        )

    def forward(
        self,
        x_continous: torch.Tensor,
        x_discrete: torch.Tensor,
        beta,
        context=None,
        context_cross=None,
    ):
        # (B, N, C) --> (B, C, N)
        x_continous = torch.permute(x_continous, (0, 2, 1)).contiguous()

        # Input projection.
        x_continous_embedding = self.continous_embedding_conv(
            x_continous
        )  # Shape (B, init_emb_dim, num_objects)
        x_discrete_embedding = self.object_model_emb(
            x_discrete
        )  # Shape (B, num_objects, init_emb_dim)
        x_discrete_embedding = torch.permute(
            x_discrete_embedding, (0, 2, 1)
        ).contiguous()

        # x has shape (B, emb_dim, num_objects).
        if self.concatenate_input_features:
            x = torch.concat(
                [x_continous_embedding, x_discrete_embedding], dim=1
            ).contiguous()
        else:
            x = x_continous_embedding + x_discrete_embedding

        ## Start: Copied from original implementation:

        if self.seperate_all:
            x_class = self.class_embedf(
                x[:, self.bbox_dim : self.bbox_dim + self.class_dim, :]
            )
            if self.objectness_dim > 0:
                x_object = self.objectness_embedf(
                    x[
                        :,
                        self.bbox_dim
                        + self.class_dim : self.bbox_dim
                        + self.class_dim
                        + self.objectness_dim,
                        :,
                    ]
                )
            else:
                x_object = 0

            if self.objfeat_dim > 0:
                x_objfeat = self.objfeat_embedf(
                    x[
                        :,
                        self.bbox_dim
                        + self.class_dim
                        + self.objectness_dim : self.bbox_dim
                        + self.class_dim
                        + self.objectness_dim
                        + self.objfeat_dim,
                        :,
                    ]
                )
            else:
                x_objfeat = 0

            x_bbox = self.bbox_embedf(x[:, 0 : self.bbox_dim, :])
            x = x_class + x_bbox + x_object + x_objfeat

        # denosing
        if context_cross is not None:
            # [B, N, C] --> [B, C, N]
            context_cross = torch.permute(context_cross, (0, 2, 1)).contiguous()

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(beta)

        h = []

        # unet-1D
        for block0, block1, attncross, block2, attn, downsample in self.downs:
            x = block0(x, context)
            x = block1(x, t)
            h.append(x)

            x = attncross(x, context_cross) if self.text_condition else attncross(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block0(x, context)
        x = self.mid_block1(x, t)
        x = (
            self.mid_attn_cross(x, context_cross)
            if self.text_condition
            else self.mid_attn_cross(x)
        )
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block0, block1, attncross, block2, attn, upsample in self.ups:
            x = block0(x, context)
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = (
                attncross(x, context_cross)
                if self.text_condition
                else self.mid_attn_cross(x)
            )
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)

        if self.seperate_all:
            out_bbox = self.bbox_hidden2output(x)
            out_class = self.class_hidden2output(x)
            out = torch.cat([out_bbox, out_class], dim=1).contiguous()
            if self.objectness_dim > 0:
                out_object = self.objectness_hidden2output(x)
                out = torch.cat([out, out_object], dim=1).contiguous()

            if self.objfeat_dim > 0:
                out_objfeat = self.objfeat_hidden2output(x)
                out = torch.cat([out, out_objfeat], dim=1).contiguous()
        else:
            out = self.final_conv(x)

        ## End: Copied from original implementation.

        denoising_output = out

        # Continous output projection.
        x_continous_out = self.to_continous_out(
            denoising_output
        )  # Shape (B, feature_dim, num_objects)

        # Discrete output projection.
        denoising_output = torch.permute(
            denoising_output, (0, 2, 1)
        )  # Shape (B, num_objects, emb_dim)
        x_discrete_out = self.to_logits(
            denoising_output
        )  # Shape (B, num_objects, max_num_objects+1)
        x_discrete_out = torch.permute(
            x_discrete_out, (0, 2, 1)
        )  # Shape (B, max_num_objects+1, num_objects)

        # Rearrange back to original shape.
        x_continous_out = torch.permute(
            x_continous_out, (0, 2, 1)
        )  # Shape (B, num_objects, feature_dim)

        return x_continous_out, x_discrete_out
