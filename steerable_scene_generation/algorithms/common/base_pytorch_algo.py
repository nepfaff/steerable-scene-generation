import pickle
import tempfile

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Union

import einops
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import yaml

from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
from PIL import Image
from pydrake.all import ModelDirectives, yaml_dump_typed


class BasePytorchAlgo(pl.LightningModule, ABC):
    """
    A base class for Pytorch algorithms using Pytorch Lightning.
    See https://lightning.ai/docs/pytorch/stable/starter/introduction.html for more details.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.debug = self.cfg.debug
        self._build_model()

    @abstractmethod
    def _build_model(self):
        """
        Create all pytorch nn.Modules here.
        """
        raise NotImplementedError

    @abstractmethod
    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        r"""Here you compute and return the training loss and some additional metrics for e.g. the progress bar or
        logger.

        Args:
            batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
            dataloader_idx: (only if multiple dataloaders used) The index of the dataloader that produced this batch.

        Return:
            Any of these options:
            - :class:`~torch.Tensor` - The loss tensor
            - ``dict`` - A dictionary. Can include any keys, but must include the key ``'loss'``.
            - ``None`` - Skip to the next batch. This is only supported for automatic optimization.
                This is not supported for multi-GPU, TPU, IPU, or DeepSpeed.

        In this step you'd normally do the forward pass and calculate the loss for a batch.
        You can also do fancier things like multiple forward passes or something model specific.

        Example::

            def training_step(self, batch, batch_idx):
                x, y, z = batch
                out = self.encoder(x)
                loss = self.loss(out, x)
                return loss

        To use multiple optimizers, you can switch to 'manual optimization' and control their stepping:

        .. code-block:: python

            def __init__(self):
                super().__init__()
                self.automatic_optimization = False


            # Multiple optimizers (e.g.: GANs)
            def training_step(self, batch, batch_idx):
                opt1, opt2 = self.optimizers()

                # do training_step with encoder
                ...
                opt1.step()
                # do training_step with decoder
                ...
                opt2.step()

        Note:
            When ``accumulate_grad_batches`` > 1, the loss returned here will be automatically
            normalized by ``accumulate_grad_batches`` internally.

        """
        return super().training_step(*args, **kwargs)

    def configure_optimizers(self):
        """
        Return an optimizer. If you need to use more than one optimizer, refer to
        pytorch lightning documentation:
        https://lightning.ai/docs/pytorch/stable/common/optimization.html
        """
        parameters = self.parameters()
        return torch.optim.Adam(parameters, lr=self.cfg.lr)

    def log_video(
        self,
        key: str,
        video: Union[np.ndarray, torch.Tensor],
        mean: Union[np.ndarray, torch.Tensor, Sequence, float] = None,
        std: Union[np.ndarray, torch.Tensor, Sequence, float] = None,
        fps: int = 12,
        format: str = "mp4",
    ):
        """
        Log video to wandb. WandbLogger in pytorch lightning does not support video
        logging yet, so we call wandb directly.

        Args:
            video: a numpy array or tensor, either in form (time, channel, height,
                width) or in the form (batch, time, channel, height, width). The content
                must be be in 0-255 if under dtype uint8 or [0, 1] otherwise.
            mean: optional, the mean to unnormalize video tensor, assuming unnormalized
                data is in [0, 1].
            std: optional, the std to unnormalize video tensor, assuming unnormalized
                data is in [0, 1].
            key: the name of the video.
            fps: the frame rate of the video.
            format: the format of the video. Can be either "mp4" or "gif".
        """
        if self.logger is None:
            return

        if isinstance(video, torch.Tensor):
            video = video.detach().cpu().numpy()

        expand_shape = [1] * (len(video.shape) - 2) + [3, 1, 1]
        if std is not None:
            if isinstance(std, (float, int)):
                std = [std] * 3
            if isinstance(std, torch.Tensor):
                std = std.detach().cpu().numpy()
            std = np.array(std).reshape(*expand_shape)
            video = video * std
        if mean is not None:
            if isinstance(mean, (float, int)):
                mean = [mean] * 3
            if isinstance(mean, torch.Tensor):
                mean = mean.detach().cpu().numpy()
            mean = np.array(mean).reshape(*expand_shape)
            video = video + mean

        if video.dtype != np.uint8:
            video = np.clip(video, a_min=0, a_max=1) * 255
            video = video.astype(np.uint8)

        self.logger.experiment.log(
            {
                key: wandb.Video(video, fps=fps, format=format),
            },
            step=self.global_step if self.global_step != 0 else wandb.run.step,
        )

    def log_image(
        self,
        key: str,
        image: Union[np.ndarray, torch.Tensor, Image.Image, Sequence[Image.Image]],
        mean: Union[np.ndarray, torch.Tensor, Sequence, float] = None,
        std: Union[np.ndarray, torch.Tensor, Sequence, float] = None,
        **kwargs: Any,
    ):
        """
        Log image(s) using WandbLogger.
        Args:
            key: the name of the video.
            image: a single image or a batch of images. If a batch of images, the shape
                should be (batch, channel, height, width).
            mean: optional, the mean to unnormalize image tensor, assuming unnormalized
                data is in [0, 1].
            std: optional, the std to unnormalize image tensor, assuming unnormalized
                data is in [0, 1].
            kwargs: optional, WandbLogger log_image kwargs, such as captions=xxx.
        """
        if self.logger is None:
            return

        if isinstance(image, Image.Image):
            image = [image]
        elif len(image) and not isinstance(image[0], Image.Image):
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()

            if len(image.shape) == 3:
                image = image[None]

            if image.shape[-1] > 4:
                # Assume the image is in the form (batch, channel, height, width)
                image = einops.rearrange(image, "b c h w -> b h w c")

            if std is not None:
                if isinstance(std, (float, int)):
                    std = [std] * 3
                if isinstance(std, torch.Tensor):
                    std = std.detach().cpu().numpy()
                std = np.array(std)[None, None, None]
                image = image * std
            if mean is not None:
                if isinstance(mean, (float, int)):
                    mean = [mean] * 3
                if isinstance(mean, torch.Tensor):
                    mean = mean.detach().cpu().numpy()
                mean = np.array(mean)[None, None, None]
                image = image + mean

            if image.dtype != np.uint8:
                image = np.clip(image, a_min=0.0, a_max=1.0) * 255
                image = image.astype(np.uint8)
                image = [img for img in image]

        if not isinstance(image, list):
            image = [img for img in image]

        self.logger.log_image(key=key, images=image, **kwargs)

    def log_html(self, key: str, html: Union[str, Sequence[str]]):
        """
        Log HTML(s) to WandB. WandbLogger in pytorch lightning does not support HTML
        logging yet, so we call wandb directly.
        Args:
            key: The name of the HTML.
            html: A single HTML or batch of HTMLs.
        """
        if self.logger is None:
            return

        if isinstance(html, str):
            wandb_html = wandb.Html(html)
        else:
            wandb_html = [wandb.Html(h) for h in html]
        self.logger.experiment.log(
            {
                key: wandb_html,
            },
            step=self.global_step if self.global_step != 0 else wandb.run.step,
        )

    def log_yaml(self, name: str, yaml_dict: Union[dict, Sequence[dict]]):
        """
        Log a dictionary to WandB as a YAML file.
        Args:
            name: The name of the YAML file.
            yaml_dict: A single dictionary or a batch of dictionaries.
        """
        if self.logger is None:
            return

        # Create a temporary YAML file and save the dictionary to it.
        if isinstance(yaml_dict, dict):
            yaml_dict = [yaml_dict]

        step = self.global_step if self.global_step != 0 else wandb.run.step
        for i, y_dict in enumerate(yaml_dict):
            with tempfile.NamedTemporaryFile(
                "w", prefix=f"{name}_{step}_{i}__", suffix=".yaml", delete=False
            ) as temp_file:
                yaml.dump(y_dict, temp_file)
                temp_file_name = temp_file.name

            self.logger.experiment.save(temp_file_name)

    def log_drake_directives(
        self, name: str, drake_directives: Sequence[ModelDirectives]
    ):
        """
        Log a dictionary to WandB as a YAML file.
        Args:
            name: The name of the YAML file.
            yaml_dict: A single dictionary or a batch of dictionaries.
        """
        if self.logger is None:
            return

        step = self.global_step if self.global_step != 0 else wandb.run.step
        for i, drake_directive in enumerate(drake_directives):
            with tempfile.NamedTemporaryFile(
                "w", prefix=f"{name}_{step}_{i}__", suffix=".dmd.yaml", delete=False
            ) as temp_file:
                temp_file_name = temp_file.name
                yaml_dump_typed(
                    drake_directive, filename=temp_file_name, schema=ModelDirectives
                )

            self.logger.experiment.save(temp_file_name)

    def log_pickle(self, name: str, obj: Any):
        """
        Log a pickle file to WandB.
        Args:
            name: The name of the pickle file.
            obj: The object to be pickled.
        """
        if self.logger is None:
            return

        step = self.global_step if self.global_step != 0 else wandb.run.step
        with tempfile.NamedTemporaryFile(
            "wb", prefix=f"{name}_{step}__", suffix=".pkl", delete=False
        ) as temp_file:
            pickle.dump(obj, temp_file)
            temp_file_name = temp_file.name

        self.logger.experiment.save(temp_file_name)

    def log_line_plot(
        self,
        key: str,
        x: Union[Sequence, torch.Tensor],
        y: Union[Sequence, torch.Tensor],
        title: Optional[str] = None,
    ):
        """
        Log a line plot to WandB. WandbLogger in pytorch lightning does not support line
        plot logging yet, so we call wandb directly.
        Args:
            key: The name of the line plot.
            x: The x-axis values.
            y: The y-axis values.
            title: The title of the line plot.
        """
        if self.logger is None:
            return

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        data = [[x, y] for x, y in zip(x, y, strict=True)]
        table = wandb.Table(data=data, columns=["x", "y"])
        self.logger.experiment.log(
            {key: wandb.plot.line(table=table, x="x", y="y", title=title)}
        )

    def log_scatter_plot(
        self,
        key: str,
        data_dict: Dict[str, Sequence],
        x_label: str,
        y_label: str,
        title: Optional[str] = None,
    ):
        """
        Log a scatter plot to WandB. WandbLogger in pytorch lightning does not support
        scatter plot logging yet, so we call wandb directly.

        Args:
            key: The name of the scatter plot.
            data_dict: A dictionary of data sequences. The keys are the names of the
                data sequences and the values are the data sequences.
            x_label: The x-axis label.
            y_label: The y-axis label.
            title: The title of the scatter plot.
        """
        if self.logger is None:
            return

        # Check that all data sequences are the same length.
        data_lengths = [len(data) for data in data_dict.values()]
        if len(set(data_lengths)) != 1:
            raise ValueError("Data sequences must all be the same length.")

        x_data = range(len(next(iter(data_dict.values()))))
        for name, y_data in data_dict.items():
            plt.scatter(x_data, y_data, label=name, s=2)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()

        fig = plt.gcf()
        self.logger.experiment.log({key: wandb.Image(fig)})

        plt.close(fig)

    def log_gradient_stats(self):
        """Log gradient statistics such as the mean or std of norm."""
        if self.logger is None:
            return

        with torch.no_grad():
            grad_norms = []
            gpr = []  # gradient-to-parameter ratio
            for param in self.parameters():
                if param.grad is not None:
                    grad_norms.append(torch.norm(param.grad).item())
                    gpr.append(torch.norm(param.grad) / torch.norm(param))
            if len(grad_norms) == 0:
                return
            grad_norms = torch.tensor(grad_norms)
            gpr = torch.tensor(gpr)
            self.log_dict(
                {
                    "train/grad_norm/min": grad_norms.min(),
                    "train/grad_norm/max": grad_norms.max(),
                    "train/grad_norm/std": grad_norms.std(),
                    "train/grad_norm/mean": grad_norms.mean(),
                    "train/grad_norm/median": torch.median(grad_norms),
                    "train/gpr/min": gpr.min(),
                    "train/gpr/max": gpr.max(),
                    "train/gpr/std": gpr.std(),
                    "train/gpr/mean": gpr.mean(),
                    "train/gpr/median": torch.median(gpr),
                }
            )

    def register_data_mean_std(
        self,
        mean: Union[str, float, Sequence],
        std: Union[str, float, Sequence],
        namespace: str = "data",
    ):
        """
        Register mean and std of data as tensor buffer.

        Args:
            mean: the mean of data.
            std: the std of data.
            namespace: the namespace of the registered buffer.
        """
        for k, v in [("mean", mean), ("std", std)]:
            if isinstance(v, str):
                if v.endswith(".npy"):
                    v = torch.from_numpy(np.load(v))
                elif v.endswith(".pt"):
                    v = torch.load(v)
                else:
                    raise ValueError(f"Unsupported file type {v.split('.')[-1]}.")
            else:
                v = torch.tensor(v)
            self.register_buffer(f"{namespace}_{k}", v.float().to(self.device))
