from typing import Union

from torch.utils.data import Dataset, Sampler


class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()

    def get_sampler(self) -> Union[Sampler, None]:
        return None
