import torch
from torch.utils.data import TensorDataset, DataLoader
from hearth.data.samplers import BatchSubsequenceSampler


def test_with_built_dataloader():
    dataset = TensorDataset(torch.ones(113, 3).cumsum(dim=0) - 1, torch.arange(113))
    batches = BatchSubsequenceSampler.build_dataloader(dataset, batch_size=5, sequence_length=4)
    assert isinstance(batches, DataLoader)
    assert isinstance(batches.batch_sampler, BatchSubsequenceSampler)
    all_batches = list(batches)
    assert len(batches.batch_sampler) == len(all_batches)

    # all except last
    for x, y in all_batches[:-1]:
        assert x.shape == (5, 4, 3)  # (batch, seq_len, feats)
        assert y.shape == (5, 4)  # (batch, seq_len)
        assert (y.reshape(5, 4, 1) == x).all().item()

    # check last batch
    last_batch_x, last_batch_y = all_batches[-1]
    assert last_batch_x.shape == (3, 4, 3)  # (batch, seq_len, feats)
    assert last_batch_y.shape == (3, 4)  # (batch, seq_len)
    assert (last_batch_y.reshape(3, 4, 1) == last_batch_x).all().item()
