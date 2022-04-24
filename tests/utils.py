import torch


def random_padded_tokens(
    batch_size: int,
    min_len: int = 1,
    max_len: int = 512,
    max_token_idx: int = 10000,
) -> torch.Tensor:
    lengths = torch.randint(min_len, max_len, size=(batch_size,))
    length = lengths.max()
    out = lengths.new_zeros(batch_size, length)
    mask = torch.arange(length).repeat(batch_size).reshape(batch_size, length) <= lengths.unsqueeze(
        -1
    )
    values = torch.randint(1, max_token_idx, size=(mask.sum(),))
    out[mask] = values
    return out
