import os
import time

from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Optional, Union

import wandb

from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.loggers.wandb import (
    ModelCheckpoint,
    Tensor,
    WandbLogger,
    _scan_checkpoints,
)
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from typing_extensions import override
from wandb_osh.hooks import TriggerWandbSyncHook

if TYPE_CHECKING:
    from wandb.sdk.lib import RunDisabled
    from wandb.wandb_run import Run


class SpaceEfficientWandbLogger(WandbLogger):
    """
    A wandb logger that by default overrides artifacts to save space, instead of
    creating new versions. An expiration_days variable can be set to control how long
    older versions of artifacts are kept. By default, the best and latest versions are
    kept indefinitely, while older versions are kept for 5 days.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: _PATH = ".",
        version: Optional[str] = None,
        offline: bool = False,
        dir: Optional[_PATH] = None,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        project: Optional[str] = None,
        log_model: Union[Literal["all"], bool] = False,
        experiment: Union["Run", "RunDisabled", None] = None,
        prefix: str = "",
        checkpoint_name: Optional[str] = None,
        expiration_days: Optional[int] = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            save_dir=save_dir,
            version=version,
            offline=offline,
            dir=dir,
            id=id,
            anonymous=anonymous,
            project=project,
            log_model=log_model,
            experiment=experiment,
            prefix=prefix,
            checkpoint_name=checkpoint_name,
            **kwargs,
        )
        self.expiration_days = expiration_days
        self._last_artifacts = []

    def _fetch_logged_artifacts(self):
        api = wandb.Api()
        run = api.run(f"{self.experiment.project}/{self.experiment.id}")
        return run.logged_artifacts()

    def _delete_previous_last_artifacts(self) -> None:
        """
        Deletes previous checkpoints that were logged due to the `save_last` option.
        """
        logged_artifacts = self._fetch_logged_artifacts()
        for artifact in logged_artifacts:
            if artifact.type != "model":
                continue
            if not self._offline:
                artifact.wait()
            original_filename = artifact.metadata.get("original_filename", "")
            if original_filename == "last.ckpt":
                if len(artifact.aliases) > 0:
                    # Don't delete the last model if it has an alias.
                    continue

                # Delete artifacts whose original filename is 'last.ckpt'
                artifact.delete()

    def _delete_all_non_top_k_artifacts(
        self, checkpoint_callback: ModelCheckpoint, is_before_logging: bool = False
    ) -> None:
        logged_artifacts = self._fetch_logged_artifacts()

        # Create a list of tuples (artifact, score) for sorting.
        artifacts_with_scores = []
        count_best_last = 0
        for artifact in logged_artifacts:
            if not self._offline:
                # Ensure that the artifact is ready.
                artifact.wait()
            if artifact.type != "model":
                continue
            if "latest" in artifact.aliases or "best" in artifact.aliases:
                # Don't delete the latest or best models.
                count_best_last += 1

                if is_before_logging and "best" not in artifact.aliases:
                    # Remove the "latest" alias from the previous best model as will
                    # log a new "latest" model.
                    artifact.aliases = []
                    artifact.save()
                    self._last_artifacts.append(artifact)

                continue
            score = artifact.metadata.get("score")
            if score is not None:
                artifacts_with_scores.append((artifact, score))

        # Sort the artifacts based on their scores. Best models come first.
        artifacts_with_scores.sort(
            key=lambda x: x[1], reverse=checkpoint_callback.mode == "max"
        )

        # Delete artifacts that are not in the top-k.
        num_to_keep = checkpoint_callback.save_top_k - count_best_last
        for artifact, _ in artifacts_with_scores[num_to_keep:]:
            # Remove any aliases that prevent deletion.
            artifact.aliases = []
            artifact.save()

            artifact.delete()

        # Ensure that all other logged artifacts have the correct expiration date.
        for artifact, _ in artifacts_with_scores[:num_to_keep]:
            if artifact.ttl is None:
                artifact.ttl = timedelta(days=self.expiration_days)
                artifact.save()

    def _scan_and_log_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
        # Delete previous artefacts that were saved due to `save_last=True`.
        self._delete_previous_last_artifacts()

        # Delete all non-top-k models.
        self._delete_all_non_top_k_artifacts(
            checkpoint_callback, is_before_logging=True
        )

        # Get checkpoints to be saved with associated score.
        checkpoints = _scan_checkpoints(checkpoint_callback, self._logged_model_time)

        # Log iteratively all new checkpoints.
        artifacts = []
        for t, p, s, tag in checkpoints:
            metadata = {
                "score": s.item() if isinstance(s, Tensor) else s,
                "original_filename": Path(p).name,
                checkpoint_callback.__class__.__name__: {
                    k: getattr(checkpoint_callback, k)
                    for k in [
                        "monitor",
                        "mode",
                        "save_last",
                        "save_top_k",
                        "save_weights_only",
                        "_every_n_train_steps",
                    ]
                    # ensure it does not break if `ModelCheckpoint` args change
                    if hasattr(checkpoint_callback, k)
                },
            }
            if not self._checkpoint_name:
                self._checkpoint_name = f"model-{self.experiment.id}"

            artifact = wandb.Artifact(
                name=self._checkpoint_name, type="model", metadata=metadata
            )
            artifact.add_file(p, name="model.ckpt")
            aliases = (
                ["latest", "best"]
                if p == checkpoint_callback.best_model_path
                else ["latest"]
            )
            self.experiment.log_artifact(artifact, aliases=aliases)
            # Remember logged models - timestamp needed in case filename didn't change
            # (lastkckpt or custom name).
            self._logged_model_time[p] = t
            artifacts.append(artifact)

        # Set expiration date for all but the best and latest models.
        for artifact in self._last_artifacts:
            try:
                if not self._offline:
                    # Ensure that the artifact is ready.
                    artifact.wait()
                if "best" in artifact.aliases:
                    # Keep the best model indefinitely.
                    continue
                artifact.ttl = timedelta(days=self.expiration_days)
                artifact.save()
            except:
                # If the artifact is already deleted, pass.
                pass

        self._last_artifacts = artifacts


class OfflineWandbLogger(SpaceEfficientWandbLogger):
    """
    Wraps WandbLogger to trigger offline sync hook occasionally.
    This is useful when running on slurm clusters, many of which
    only has internet on login nodes, not compute nodes.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: _PATH = ".",
        version: Optional[str] = None,
        offline: bool = False,
        dir: Optional[_PATH] = None,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        project: Optional[str] = None,
        log_model: Union[Literal["all"], bool] = False,
        experiment: Union["Run", "RunDisabled", None] = None,
        prefix: str = "",
        checkpoint_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            save_dir=save_dir,
            version=version,
            offline=False,
            dir=dir,
            id=id,
            anonymous=anonymous,
            project=project,
            log_model=log_model,
            experiment=experiment,
            prefix=prefix,
            checkpoint_name=checkpoint_name,
            **kwargs,
        )
        self._offline = offline
        communication_dir = Path(".wandb_osh_command_dir")
        communication_dir.mkdir(parents=True, exist_ok=True)
        self.trigger_sync = TriggerWandbSyncHook(communication_dir)
        self.last_sync_time = 0.0
        self.min_sync_interval = 60
        self.wandb_dir = os.path.join(self._save_dir, "wandb/latest-run")

    @override
    @rank_zero_only
    def log_metrics(
        self, metrics: Mapping[str, float], step: Optional[int] = None
    ) -> None:
        out = super().log_metrics(metrics, step)
        if time.time() - self.last_sync_time > self.min_sync_interval:
            self.trigger_sync(self.wandb_dir)
            self.last_sync_time = time.time()
        return out
