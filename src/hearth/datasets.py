from typing import Union, Sized
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset as _TensorDataset

from hearth.containers import TensorDict
from hearth._collate import default_collate


class BatchesMixin:

    """mixin for supporting batches and collate method from datasets."""

    def collate(self, batch):
        return default_collate(batch)

    def batches(
        self,
        batch_size: int = 32,
        shuffle: bool = False,
        drop_last: bool = False,
        num_workers: int = 0,
        **kwargs,
    ) -> DataLoader:
        """return a Dataloader that iterates over batches of this dataset.

        Note:
            this method supports additional keyword args which will be passed to the dataloader.

        Args:
            batch_size: Defaults to 32.
            shuffle: if True shuffle the dataset. Defaults to False.
            drop_last: if true drop the last batch if it is less than batch size. Defaults to False.
            num_workers: number of workers. Defaults to 0.
        """
        return DataLoader(
            self,  # type: ignore
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=drop_last,
            shuffle=shuffle,
            collate_fn=self.collate,
            **kwargs,
        )


@dataclass
class XYDataset(Dataset, BatchesMixin):
    """basic dataset that returns a tuple of inputs and targets.

    supports :class:`hearth.containers.TensorDataset`
    """

    x: Union[Sized, TensorDict]
    y: Union[Sized, TensorDict]

    def __len__(self) -> int:
        if isinstance(self.x, TensorDict):
            return len(next(iter(self.x.values())))  # type: ignore
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class TensorDataset(_TensorDataset, BatchesMixin):
    pass
