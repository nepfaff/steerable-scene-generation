from pathlib import Path
from typing import Any, Dict, List

import wandb

from omegaconf.listconfig import ListConfig


def is_run_id(run_id: str) -> bool:
    """Check if a string is a run ID."""
    return len(run_id) == 8 and run_id.isalnum()


def version_to_int(artifact: wandb.Artifact) -> int:
    """Convert versions of the form vX to X. For example, v12 to 12."""
    return int(artifact.version[1:])


def download_latest_or_best_checkpoint(
    run_path: str, download_dir: Path, use_best: bool
) -> Path:
    api = wandb.Api()
    run = api.run(run_path)

    # Find the latest saved model checkpoint.
    latest = None
    best = None
    artifact: wandb.Artifact
    for artifact in run.logged_artifacts():
        if artifact.type != "model" or artifact.state != "COMMITTED":
            continue

        if latest is None or version_to_int(artifact) > version_to_int(latest):
            latest = artifact

        if "best" in artifact.aliases:
            best = artifact

    if (use_best and best is None) or latest is None:
        raise ValueError("No model checkpoints found.")
    elif use_best and best is None:
        print("Warning: No best model checkpoint found. Using latest.")

    # Download the checkpoint.
    download_dir.mkdir(exist_ok=True, parents=True)
    root = download_dir / run_path
    checkpoint = best if use_best and best is not None else latest
    checkpoint.download(root=root)
    return root / "model.ckpt"


def download_version_checkpoint(
    run_path: str, download_dir: Path, version: int
) -> Path:
    api = wandb.Api()
    run = api.run(run_path)

    # Find the checkpoint with the specified version.
    checkpoint = None
    artifact: wandb.Artifact
    for artifact in run.logged_artifacts():
        if artifact.type != "model" or artifact.state != "COMMITTED":
            continue

        if version == version_to_int(artifact):
            checkpoint = artifact
            break

    if checkpoint is None:
        raise ValueError(f"No model checkpoints with version {version} found.")

    # Download the checkpoint.
    download_dir.mkdir(exist_ok=True, parents=True)
    root = download_dir / run_path
    checkpoint.download(root=root)
    return root / "model.ckpt"


def download_all_checkpoints(
    run_path: str, download_dir: Path, versions: List[int] | ListConfig | None = None
) -> List[Dict[str, Any]]:
    """
    Download all model checkpoints from a wandb run.

    Args:
        run_path (str): The path to the wandb run.
        download_dir (Path): The directory to download the checkpoints to.
        versions (List[int] | ListConfig | None): A list of versions to download. If
            None, download all checkpoints.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing:
            - path (Path): The path to the downloaded checkpoint.
            - version (int): The version of the checkpoint.
    """
    if not (versions is None or isinstance(versions, (list, ListConfig))):
        raise ValueError(
            f"Versions must be a list of integers or None. Got {type(versions)}."
        )
    if versions is not None and len(versions) == 0:
        raise ValueError("Empty versions list provided.")

    api = wandb.Api()
    run = api.run(run_path)

    # Download all model checkpoints.
    download_dir.mkdir(exist_ok=True, parents=True)
    root = download_dir / run_path
    checkpoints = []
    artifact: wandb.Artifact
    for artifact in run.logged_artifacts():
        if artifact.type != "model" or artifact.state != "COMMITTED":
            continue

        if not versions is None and not version_to_int(artifact) in versions:
            # Skip this checkpoint.
            continue

        checkpoint = artifact
        path = root / checkpoint.name
        checkpoint.download(root=path)

        checkpoints.append(
            {
                "path": path / "model.ckpt",
                "version": version_to_int(artifact),
            }
        )

    # Sort the checkpoints by version.
    checkpoints.sort(key=lambda x: x["version"])

    return checkpoints
