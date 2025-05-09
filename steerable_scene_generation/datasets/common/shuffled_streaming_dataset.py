import random

import torch

from datasets import Dataset


class ShuffledStreamingDataset(torch.utils.data.IterableDataset):
    """
    A streaming dataset that uses a random offset and keeps a data buffer for
    shuffling.
    """

    def __init__(self, dataset: Dataset, buffer_size: int | None = 2048):
        """
        Args:
            dataset: The dataset to shuffle.
            buffer_size: The size of the buffer to use for shuffling. Set to None to
                disable shuffling.
        """
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        # Random offset at each epoch.
        indices = list(range(len(self.dataset)))
        offset = random.randint(0, len(indices) - 1)
        indices = indices[offset:] + indices[:offset]

        # Optional shuffle buffer.
        if self.buffer_size is None:
            for i in indices:
                yield self.dataset[i]
        else:
            buffer = []
            for i in indices:
                item = self.dataset[i]
                buffer.append(item)
                if len(buffer) >= self.buffer_size:
                    random.shuffle(buffer)
                    for x in buffer:
                        yield x
                    buffer.clear()
            # Yield the remaining items that didn't fill the buffer.
            random.shuffle(buffer)
            for x in buffer:
                yield x


class InfiniteShuffledStreamingDataset(ShuffledStreamingDataset):
    """
    A streaming dataset that shuffles the data at each epoch and uses a random
    offset at each epoch.
    The iterator is re-started after it's exhausted.
    """

    def __init__(self, dataset: Dataset, buffer_size: int | None = 2048):
        """
        Args:
            dataset: The dataset to shuffle.
            buffer_size: The size of the buffer to use for shuffling. Set to None to
                disable shuffling.
        """
        super().__init__(dataset=dataset, buffer_size=buffer_size)

    def __iter__(self):
        while True:
            try:
                yield from super().__iter__()
            except StopIteration:
                # Reset the dataset iterator.
                continue
