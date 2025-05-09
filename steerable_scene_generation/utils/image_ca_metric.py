"""
Code for computing the CA (classifier accuracy) metric between dataset and synthesized
scene semantic images.

Adapted from
https://github.com/MIT-SPARK/ThreedFront/blob/main/scripts/synthetic_vs_real_classifier.py
"""

import os

from typing import List, Tuple, Union

import numpy as np
import torch

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from steerable_scene_generation.utils.classifiers import compute_loss_and_acc


class ImageFolderDataset(torch.utils.data.Dataset):
    """
    Dataset class for a directory of png images. Half the images are used for training
    and half are used for testing.
    """

    def __init__(self, directory: str, seed: int, train: bool = True):
        image_paths = sorted(
            [
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if f.endswith("png")
            ]
        )
        # Shuffle.
        np.random.seed(seed)
        np.random.shuffle(image_paths)

        N = len(image_paths) // 2
        start = 0 if train else N
        self.image_paths = image_paths[start : start + N]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.image_paths[idx]


class SyntheticVRealDataset(torch.utils.data.Dataset):
    """Dataset combining preprocessed real images and synthetic images"""

    def __init__(
        self,
        real_dataset: ImageFolderDataset,
        synthetic_dataset: ImageFolderDataset,
    ):
        self.real_dataset_length = len(real_dataset)
        self.real = real_dataset
        self.synthetic = synthetic_dataset

    def __len__(self):
        return len(self.real) + len(self.synthetic)

    def __getitem__(self, idx):
        if idx < self.real_dataset_length:
            image_path = self.real[idx]
            label = 1
        else:
            image_path = self.synthetic[idx - self.real_dataset_length]
            label = 0

        img = Image.open(image_path)
        img = np.asarray(img).astype(np.float32) / np.float32(255)  # Shape (W, H)
        img = np.transpose(img[:, :, :3], (2, 0, 1))  # Shape (C, W, H)

        return torch.from_numpy(img), torch.tensor([label], dtype=torch.float32)


class AlexNet(torch.nn.Module):
    """Modified AlexNet to distinguish real and synthetic images."""

    def __init__(self):
        super().__init__()

        self.model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        self.fc = torch.nn.Linear(9216, 1)

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = self.fc(x.view(len(x), -1))
        x = torch.sigmoid(x)

        return x


def get_score(
    dataloader: DataLoader, model: torch.nn.Module, device: torch.device
) -> float:
    """Calculates the score based on the accuracy of the model.

    Args:
        dataloader (DataLoader): The DataLoader containing the dataset.
        model (torch.nn.Module): The model used for evaluation.
        device (torch.device): The device on which the model is located.

    Returns:
        float: The computed score based on the model's accuracy.
    """
    _, acc = compute_loss_and_acc(dataloader, model, device)
    score = acc * 100
    if score < 50.0:
        # Avoid <50% runs pulling the mean towards 50%.
        score = 100.0 - score
    return score


@torch.enable_grad()
def compute_image_ca_metric(
    dataset_directory: str,
    synthesized_directory: str,
    batch_size: int = 256,
    num_workers: int = 4,
    iterations: Union[int, List[int]] = 100,
    num_runs: int = 10,
    seed: int = 42,
) -> Union[Tuple[float, float], List[Tuple[float, float]]]:
    """
    Compute the classifier accuracy metric between the dataset and synthetic images.
    An accuracy of 50% is the best score, meaning that the classifier is unable to
    differentiate between the two image sets.

    Args:
        dataset_directory (str): Path to directory containing the dataset semantic
            images.
        synthesized_directory (str): Path to the directory containing the synthesized
            semantic images.
        batch_size (int): The batch size.
        num_workers (int): The number of data loader workers.
        iterations (Union[int, List[int]]): The target number of iterations to train the classifier for.
            If a list is provided, then the results are returned for each iteration count in the list.
        num_runs (int): The number of runs to average the results over.
        seed (int): The random seed.

    Return:
        Union[Tuple[float, float], List[Tuple[float, float]]]:
            - If `iterations` is an int: A tuple of (mean_accuracy, std_accuracy).
            - If `iterations` is a List[int]: A list of (mean_accuracy, std_accuracy)
                tuples, of the same length as the `iterations` list.
    """
    # Set random seed.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    scores = []
    for run in tqdm(range(num_runs), desc="Training runs", position=0, leave=False):
        # Create the datasets.
        random_seed = np.random.randint(0, 2**32)
        train_real = ImageFolderDataset(dataset_directory, seed=random_seed, train=True)
        test_real = ImageFolderDataset(dataset_directory, seed=random_seed, train=False)
        train_synthetic = ImageFolderDataset(
            synthesized_directory, seed=random_seed, train=True
        )
        test_synthetic = ImageFolderDataset(
            synthesized_directory, seed=random_seed, train=False
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
        model = AlexNet()
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Prepare for iteration tracking.
        max_iterations = iterations if isinstance(iterations, int) else max(iterations)
        iteration_scores = (
            {iter_count: [] for iter_count in iterations}
            if isinstance(iterations, list)
            else []
        )

        # Train the model.
        model = model.train()
        train_iter = iter(train_dataloader)
        for iteration in tqdm(
            range(max_iterations), desc="    Iterations", position=1, leave=False
        ):
            # Get a new batch, cycling through the dataset as needed.
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                x, y = next(train_iter)

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            y_hat = model(x)
            loss = torch.nn.functional.binary_cross_entropy(y_hat, y)

            loss.backward()
            optimizer.step()

            # Check if we need to evaluate at this iteration.
            if isinstance(iterations, list) and iteration + 1 in iterations:
                # Evaluate.
                score = get_score(test_dataloader, model, device)
                iteration_scores[iteration + 1].append(score)
                print(f"run {run}, iteration {iteration+1} accuracy: {score:.4f}")

                train_score = get_score(train_dataloader, model, device)
                print(
                    f"run {run}, iteration {iteration+1} train accuracy: "
                    f"{train_score:.4f}"
                )

        # Evaluate at the end if using a single iteration count.
        if isinstance(iterations, int):
            score = get_score(test_dataloader, model, device)
            scores.append(score)
            print(f"run {run} accuracy: {score:.4f}")

            train_score = get_score(train_dataloader, model, device)
            print(f"run {run} train accuracy: {train_score:.4f}")
        else:
            scores.append(iteration_scores)

    if isinstance(iterations, int):
        return np.mean(scores), np.std(scores)

    # Aggregate the results for each iteration count in the list.
    mean_std_results = []
    for iter_count in iterations:
        iter_scores_list = [run_scores[iter_count] for run_scores in scores]
        mean_accuracy = np.mean(iter_scores_list)
        std_accuracy = np.std(iter_scores_list)
        mean_std_results.append((mean_accuracy, std_accuracy))
    return mean_std_results
