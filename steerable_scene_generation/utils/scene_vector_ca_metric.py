"""
Code for computing the CA (classifier accuracy) metric between dataset and synthesized
scene vectors.
"""

from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from steerable_scene_generation.utils.classifiers import compute_loss_and_acc


class DeepSetsSceneVectorClassifier(nn.Module):
    """
    A set classifier for scenes that maintains permutation invariance accross the object
    dimension.
    The architecture is inspired by DeepSets: https://arxiv.org/abs/1703.06114.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()

        # Phi network applied to each object independently.
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Rho network applied after object aggregation.
        self.rho = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, V = x.size()

        # Apply phi to each object independently.
        x = x.view(B * N, V)  # Shape (B * N, V)
        x = self.phi(x)
        x = x.view(B, N, -1)  # Shape (B, N, hidden_dim)

        # Mean and max pooling (permutation-invariant).
        x_mean = x.mean(dim=1)  # Shape (B, hidden_dim)
        x_max = x.max(dim=1).values  # Shape (B, hidden_dim)
        x = torch.cat([x_mean, x_max], dim=-1)  # Shape (B, 2*hidden_dim)

        # Apply rho to the aggregated representation.
        x = self.rho(x)  # Shape (B, 1)
        x = torch.sigmoid(x)  # Shape (B, 1)
        return x


class TransformerSceneVectorClassifier(nn.Module):
    """
    A transformer-based classifier for scene vectors.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        """
        Args:
            input_dim (int): The dimension of the input scene vectors.
            hidden_dim (int): The dimension of the hidden state of the transformer.
            num_layers (int): The number of transformer layers.
            num_heads (int): The number of attention heads in the transformer.
            dropout (float): The dropout rate for the transformer.
        """
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Learnable class token.
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Transformer encoder stack.
        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,  # Pre-norm (ViT-style)
                )
                for _ in range(num_layers)
            ]
        )

        # Classification head.
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input scene vectors of shape (B, N, D).

        Returns:
            torch.Tensor: The output of the classifier of shape (B, 1).
        """
        # Project input, independently for each object.
        x = self.input_proj(x)

        # Append class token.
        cls = self.cls_token.expand(x.shape[0], -1, -1)  # (B, 1, D)
        x = torch.cat([cls, x], dim=1)  # (B, 1+N, D)

        # Transformer layers.
        for layer in self.encoder_layers:
            x = layer(x)

        # Extract cls token.
        x_cls = x[:, 0, :]  # (B, D)

        # Binary classification.
        return torch.sigmoid(self.head(x_cls))


class SceneDataset(torch.utils.data.Dataset):
    """
    Dataset class for a set of scenes. Half the scenes are used for training and half
    are used for testing.
    """

    def __init__(self, scenes: torch.Tensor, seed: int, train: bool = True):
        # Shuffle.
        torch.manual_seed(seed)
        indices = torch.randperm(len(scenes))
        scenes = scenes[indices]

        # Use the first half for training and the second half for testing.
        N = len(scenes) // 2
        start = 0 if train else N
        self.scenes = scenes[start : start + N]

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        return self.scenes[idx]


class SyntheticVRealDataset(torch.utils.data.Dataset):
    """Dataset combining preprocessed real and synthetic scenes."""

    def __init__(
        self,
        real_dataset: SceneDataset,
        synthetic_dataset: SceneDataset,
    ):
        self.real_dataset_length = len(real_dataset)
        self.real = real_dataset
        self.synthetic = synthetic_dataset

    def __len__(self):
        return len(self.real) + len(self.synthetic)

    def __getitem__(self, idx):
        if idx < self.real_dataset_length:
            scene = self.real[idx]
            label = 1
        else:
            scene = self.synthetic[idx - self.real_dataset_length]
            label = 0

        return scene, torch.tensor([label], dtype=torch.float32)


@torch.enable_grad()
def compute_scene_vector_ca_metric(
    dataset_scenes: torch.Tensor,
    synthesized_scenes: torch.Tensor,
    batch_size: int = 128,
    num_workers: int = 4,
    epochs: Union[int, List[int]] = 10,
    num_runs: int = 10,
    use_transformer: bool = False,
    seed: int = 42,
) -> Union[Tuple[float, float], List[Tuple[float, float]]]:
    """
    Compute the classifier accuracy metric between the dataset and synthetic scene
    vectors.
    An accuracy of 50% is the best score, meaning that the classifier is unable to
    differentiate between the two scene sets.

    Args:
        dataset_scenes (torch.Tensor): The dataset scenes of shape (B, N, V).
        synthesized_scenes (torch.Tensor): The synthesized scenes of shape (B, N, V).
        batch_size (int): The batch size.
        num_workers (int): The number of data loader workers.
        epochs (int): The number of epochs to finetune the classifier for. If a list is
            provided, then the results are returned for each epoch in the list.
        num_runs (int): The number of runs to average the results over.
        use_transformer (bool): Whether to use the classifier based on the
            Set Transformer architecture. Defaults to the DeepSets architecture.
        seed (int): The random seed.

    Return:
        Union[Tuple[float, float], List[Tuple[float, float]]]:
            - If `epochs` is an int: A tuple of (mean_accuracy, std_accuracy).
            - If `epochs` is a List[int]: A list of (mean_accuracy, std_accuracy)
                tuples, of the same length as the `epochs` list.
    """
    assert (
        dataset_scenes.shape[-1] == synthesized_scenes.shape[-1]
        and dataset_scenes.shape[-2] == synthesized_scenes.shape[-2],
        "Check that the datset matches!",
    )

    # Set random seed.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def get_score(dataloader, model, device):
        _, acc = compute_loss_and_acc(dataloader, model, device)
        score = acc * 100
        if score < 50.0:
            # Avoid <50% runs pulling the mean towards 50%.
            score = 100.0 - score
        return score

    scores = []
    for run in tqdm(range(num_runs), desc="Training runs", position=0, leave=False):
        # Create the datasets.
        random_seed = np.random.randint(0, 2**32)
        train_real = SceneDataset(scenes=dataset_scenes, seed=random_seed, train=True)
        test_real = SceneDataset(scenes=dataset_scenes, seed=random_seed, train=False)
        train_synthetic = SceneDataset(
            scenes=synthesized_scenes, seed=random_seed, train=True
        )
        test_synthetic = SceneDataset(
            scenes=synthesized_scenes, seed=random_seed, train=False
        )

        # Join them in useable datasets.
        train_dataset = SyntheticVRealDataset(
            real_dataset=train_real, synthetic_dataset=train_synthetic
        )
        test_dataset = SyntheticVRealDataset(
            real_dataset=test_real, synthetic_dataset=test_synthetic
        )

        # Create the dataloaders.
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
        )
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
        )

        # Create the model.
        if use_transformer:
            model = TransformerSceneVectorClassifier(input_dim=dataset_scenes.shape[-1])
        else:
            model = DeepSetsSceneVectorClassifier(input_dim=dataset_scenes.shape[-1])
        model = model.to(device)

        # Calculate total training steps.
        num_epochs = epochs if isinstance(epochs, int) else epochs[-1]
        total_steps = num_epochs * len(train_dataloader)
        warmup_steps = int(0.05 * total_steps)  # 5% warmup

        # Optimizer and scheduler.
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # Train the model.
        model = model.train()
        epoch_scores = (
            {epoch: [] for epoch in epochs} if isinstance(epochs, list) else []
        )
        for epoch in tqdm(
            range(num_epochs), desc="    Epochs", position=1, leave=False
        ):
            for x, y in train_dataloader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                y_hat = model(x)
                loss = torch.nn.functional.binary_cross_entropy(y_hat, y)

                loss.backward()
                optimizer.step()
                scheduler.step()

            if isinstance(epochs, list) and epoch + 1 in epochs:
                # Evaluate.
                score = get_score(test_dataloader, model, device)
                epoch_scores[epoch + 1].append(score)
                print(f"run {run}, epoch {epoch+1} accuracy: {score:.4f}")

                train_score = get_score(train_dataloader, model, device)
                print(f"run {run}, epoch {epoch+1} train accuracy: {train_score:.4f}")

        # Evaluate.
        if isinstance(epochs, int):
            score = get_score(test_dataloader, model, device)
            scores.append(score)
            print(f"run {run} accuracy: {score:.4f}")

            train_score = get_score(train_dataloader, model, device)
            print(f"run {run} train accuracy: {train_score:.4f}")
        else:
            scores.append(epoch_scores)

    if isinstance(epochs, int):
        return np.mean(scores), np.std(scores)

    # Aggregate the results for each epoch in the list
    mean_std_results = []
    for epoch in epochs:
        epoch_scores_list = [run_scores[epoch] for run_scores in scores]
        mean_accuracy = np.mean(epoch_scores_list)
        std_accuracy = np.std(epoch_scores_list)
        mean_std_results.append((mean_accuracy, std_accuracy))
    return mean_std_results
