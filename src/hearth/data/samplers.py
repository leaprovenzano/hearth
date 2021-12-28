"""Samplers for sampling batch indices (or example indexes) for use with \
`torch.utils.data.DataLoader\
 <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_
"""
from typing import Sized, List, Iterator
import torch
from torch.utils.data import Sampler, DataLoader

from more_itertools import windowed


class SubsequenceSampler(Sampler):
    """a batch sampler that keeps sequences aligned within batches but shuffles batches.

    Often when training on a dataset that represents a full sequence we may want to
    window that sequence into multiple subsequences and shuffle those. Additionally
    when batch_size is not divisible by dataset length we choose a different short sequence
    on each iteration which means batches are not always starting and ending at the same
    position. The shortest batch will always be last to comply with expected behavior from
    other batchers.

    Args:
        dataset: something we can call ``len()`` on that represents the size of the dataset being
            batched. It can be the dataset itself, a tensor, list, range etc...
        batch_size: the desired batch size
        drop_shortest: if ``True`` drop the shortest batch on each iteration if the size of the
            dataset is not divisible by ``batch_size``. The short batch will be chosen randomly
            on each iteration (providing a little extra noise and ensuring we dont see exactly
            the same subsequences on each iteration). Defaults to False.

    Example:

        >>> import torch
        >>> from hearth.data.samplers import SubsequenceSampler
        >>> _ = torch.manual_seed(0)
        >>>
        >>> # subsequence sampler only dataset
        >>> # so it could be a range object or a Dataset, a tensor etc...
        >>> sampler = SubsequenceSampler(range(15), batch_size=4)
        >>> sampler
        SubsequenceSampler(dataset=range(0, 15), batch_size=4, drop_shortest=False)

        by default we will get one short batch of 3, since ``batch_size=4`` and  \
        ``drop_shortest=False``:

        >>> for batch in sampler:
        ...     print(batch)
        [11, 12, 13, 14]
        [7, 8, 9, 10]
        [3, 4, 5, 6]
        [0, 1, 2]


        between runs we shuffle and choose a new short batch:

        >>> for batch in sampler:
        ...     print(batch)
        [8, 9, 10, 11]
        [0, 1, 2, 3]
        [4, 5, 6, 7]
        [12, 13, 14]


        when ``drop_shortest=True`` the randomly chosen short batch will be dropped...

        >>> for batch in SubsequenceSampler(range(15), batch_size=4, drop_shortest=True):
        ...     print(batch)
        [11, 12, 13, 14]
        [0, 1, 2, 3]
        [7, 8, 9, 10]

        and again this will be different for each iteration:

        >>> for batch in SubsequenceSampler(range(15), batch_size=4, drop_shortest=True):
        ...     print(batch)
        [11, 12, 13, 14]
        [7, 8, 9, 10]
        [3, 4, 5, 6]


        when ``batch_size`` is divisible by the number of examples in the dataset
        behavior is a little more deterministic, since the indexes in each batch will always
        be the same:

        >>> for batch in SubsequenceSampler(range(12), batch_size=4):
        ...     print(batch)
        [0, 1, 2, 3]
        [4, 5, 6, 7]
        [8, 9, 10, 11]

        but batches will still be shuffled between iterations:

        >>> for batch in SubsequenceSampler(range(12), batch_size=4):
        ...     print(batch)
        [8, 9, 10, 11]
        [0, 1, 2, 3]
        [4, 5, 6, 7]

    """

    def __init__(self, dataset: Sized, batch_size: int, drop_shortest: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_shortest = drop_shortest

    def _get_lengths(self, total_samples: int) -> List[int]:
        short_batch_size = total_samples % self.batch_size
        n_batches = total_samples // self.batch_size + (short_batch_size > 0)
        lengths = [self.batch_size] * n_batches

        if short_batch_size != 0:
            short_batch_idx: int = torch.randint(high=n_batches, size=(1,)).item()  # type: ignore
            lengths[short_batch_idx] = short_batch_size

        return lengths

    def _batches(self) -> Iterator[List[int]]:
        total_samples = len(self.dataset)
        batches = torch.split_with_sizes(
            torch.arange(total_samples), self._get_lengths(total_samples)
        )
        sort_keys = torch.randperm(len(batches)).tolist()
        # here we ensure that the shortest batch is last
        yield from sorted([batches[i].tolist() for i in sort_keys], key=len, reverse=True)

    def __iter__(self) -> Iterator[List[int]]:
        for batch in self._batches():
            if not self.drop_shortest or (len(batch) == self.batch_size):
                yield batch

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_shortest:
            return n // self.batch_size  # type: ignore
        else:
            return (n + self.batch_size - 1) // self.batch_size

    def __repr__(self) -> str:
        name = self.__class__.__name__
        args = (
            f'dataset={self.dataset},'
            f' batch_size={self.batch_size},'
            f' drop_shortest={self.drop_shortest}'
        )
        return f'{name}({args})'


class BatchSubsequenceSampler(torch.utils.data.Sampler):
    """a batch sampler that generates nested ordered subsequence indexes aligned on batch.

    output indexes from this sampler will be  lists of shape [batch_size, sequence_lengh].

    Note:
        Internally this uses :class:`SubsequenceSampler` to generate batches and will always drop
        the short sequence if the length of the dataset is not divisible by `sequence_length`.

    Args:
        dataset: something we can call ``len()`` on that represents the size of the dataset being
            batched. It can be the dataset itself, a tensor, list, range etc...
        batch_size: the desired batch size
        sequence_length: desired sequence length
        drop_last: if ``True`` drop the batch if less than batch size.

    Example:
        >>> import torch
        >>> from hearth.data.samplers import SubsequenceSampler
        >>> _ = torch.manual_seed(0)
        >>> sampler = BatchSubsequenceSampler(range(113), batch_size=4, sequence_length=5)
        >>> len(sampler)
        6

        generate batch indexes for sequental sequences of shape (`batch_size`, `sequence_length`)
        the last batch may contain less sequences, but sequence lengths will never change.

        >>> for batch_idxes in sampler:
        ...    print(batch_idxes)
        [[20, 21, 22, 23, 24], [10, 11, 12, 13, 14], [55, 56, 57, 58, 59], [30, 31, 32, 33, 34]]
        [[5, 6, 7, 8, 9], [60, 61, 62, 63, 64], [80, 81, 82, 83, 84], [15, 16, 17, 18, 19]]
        [[0, 1, 2, 3, 4], [90, 91, 92, 93, 94], [103, 104, 105, 106, 107], [40, 41, 42, 43, 44]]
        [[85, 86, 87, 88, 89], [50, 51, 52, 53, 54], [35, 36, 37, 38, 39], [75, 76, 77, 78, 79]]
        [[45, 46, 47, 48, 49], [65, 66, 67, 68, 69], [70, 71, 72, 73, 74], [108, 109, 110, 111, 112]]
        [[98, 99, 100, 101, 102], [25, 26, 27, 28, 29]]
    """  # noqa : E501

    @classmethod
    def build_dataloader(
        cls,
        dataset: Sized,
        batch_size: int,
        sequence_length: int,
        drop_last: bool = False,
        **kwargs,
    ) -> DataLoader:
        """creates a new DataLoader using BatchSubsequenceSampler.

        Note:
            extra keyword arguments will be passed to \
            `torch.utils.data.DataLoader\
                <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_

        Args:
            dataset: actual dataset you'd like to use, this dataset should support nested
                multi-indexing, `torch.utils.dataset.TensorDataset` works just fine
            batch_size: the desired batch size for the sequences
            sequence_length: desired sequence length for subsequences
            drop_last: If true drop the last short batch. Defaults to True.
        """
        n_samples = len(dataset)
        # TODO: check dataset to ensure we can tolerate this kind of batch
        sampler = cls(
            range(n_samples),
            batch_size=batch_size,
            sequence_length=sequence_length,
            drop_last=drop_last,
        )
        return DataLoader(dataset, batch_sampler=sampler, **kwargs)  # type: ignore

    def __init__(
        self, dataset: Sized, batch_size: int, sequence_length: int, drop_last: bool = False
    ):
        self.subseq_sampler = SubsequenceSampler(
            dataset, batch_size=sequence_length, drop_shortest=True
        )
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __len__(self) -> int:
        n = len(self.subseq_sampler)
        return (n // self.batch_size) + ((n % self.batch_size) > 0 and not self.drop_last)

    def __iter__(self) -> Iterator[List[List[int]]]:
        for sequences in windowed(self.subseq_sampler, self.batch_size, step=self.batch_size):
            seq_batch: List[List[int]] = list(filter(lambda x: x, sequences))  # type: ignore
            if len(seq_batch) == self.batch_size or not self.drop_last:
                yield seq_batch
