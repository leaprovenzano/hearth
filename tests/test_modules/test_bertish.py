import pytest
import torch
from hearth.modules import Bertish
from tests.utils import random_padded_tokens


@pytest.mark.parametrize('model,', [Bertish()])
def test_torchscriptability(model):
    tokens = random_padded_tokens(5)
    model.eval()
    with torch.no_grad():
        pyout = model(tokens)

    try:
        scripted = model.script()
        with torch.no_grad():
            scripted_out = scripted(tokens)
    except RuntimeError as err:
        pytest.fail(f'failed to script model with script method! raised {err}')
    assert scripted_out.shape == pyout.shape
    torch.testing.assert_allclose(scripted_out, pyout)


def test_load_transformers_bert_state_dict():
    layout = {
        'embeddings.position_ids': torch.Size([1, 512]),
        'embeddings.word_embeddings.weight': torch.Size([30522, 256]),
        'embeddings.position_embeddings.weight': torch.Size([512, 256]),
        'embeddings.token_type_embeddings.weight': torch.Size([2, 256]),
        'embeddings.LayerNorm.weight': torch.Size([256]),
        'embeddings.LayerNorm.bias': torch.Size([256]),
        'encoder.layer.0.attention.self.query.weight': torch.Size([256, 256]),
        'encoder.layer.0.attention.self.query.bias': torch.Size([256]),
        'encoder.layer.0.attention.self.key.weight': torch.Size([256, 256]),
        'encoder.layer.0.attention.self.key.bias': torch.Size([256]),
        'encoder.layer.0.attention.self.value.weight': torch.Size([256, 256]),
        'encoder.layer.0.attention.self.value.bias': torch.Size([256]),
        'encoder.layer.0.attention.output.dense.weight': torch.Size([256, 256]),
        'encoder.layer.0.attention.output.dense.bias': torch.Size([256]),
        'encoder.layer.0.attention.output.LayerNorm.weight': torch.Size([256]),
        'encoder.layer.0.attention.output.LayerNorm.bias': torch.Size([256]),
        'encoder.layer.0.intermediate.dense.weight': torch.Size([1024, 256]),
        'encoder.layer.0.intermediate.dense.bias': torch.Size([1024]),
        'encoder.layer.0.output.dense.weight': torch.Size([256, 1024]),
        'encoder.layer.0.output.dense.bias': torch.Size([256]),
        'encoder.layer.0.output.LayerNorm.weight': torch.Size([256]),
        'encoder.layer.0.output.LayerNorm.bias': torch.Size([256]),
        'encoder.layer.1.attention.self.query.weight': torch.Size([256, 256]),
        'encoder.layer.1.attention.self.query.bias': torch.Size([256]),
        'encoder.layer.1.attention.self.key.weight': torch.Size([256, 256]),
        'encoder.layer.1.attention.self.key.bias': torch.Size([256]),
        'encoder.layer.1.attention.self.value.weight': torch.Size([256, 256]),
        'encoder.layer.1.attention.self.value.bias': torch.Size([256]),
        'encoder.layer.1.attention.output.dense.weight': torch.Size([256, 256]),
        'encoder.layer.1.attention.output.dense.bias': torch.Size([256]),
        'encoder.layer.1.attention.output.LayerNorm.weight': torch.Size([256]),
        'encoder.layer.1.attention.output.LayerNorm.bias': torch.Size([256]),
        'encoder.layer.1.intermediate.dense.weight': torch.Size([1024, 256]),
        'encoder.layer.1.intermediate.dense.bias': torch.Size([1024]),
        'encoder.layer.1.output.dense.weight': torch.Size([256, 1024]),
        'encoder.layer.1.output.dense.bias': torch.Size([256]),
        'encoder.layer.1.output.LayerNorm.weight': torch.Size([256]),
        'encoder.layer.1.output.LayerNorm.bias': torch.Size([256]),
        'encoder.layer.2.attention.self.query.weight': torch.Size([256, 256]),
        'encoder.layer.2.attention.self.query.bias': torch.Size([256]),
        'encoder.layer.2.attention.self.key.weight': torch.Size([256, 256]),
        'encoder.layer.2.attention.self.key.bias': torch.Size([256]),
        'encoder.layer.2.attention.self.value.weight': torch.Size([256, 256]),
        'encoder.layer.2.attention.self.value.bias': torch.Size([256]),
        'encoder.layer.2.attention.output.dense.weight': torch.Size([256, 256]),
        'encoder.layer.2.attention.output.dense.bias': torch.Size([256]),
        'encoder.layer.2.attention.output.LayerNorm.weight': torch.Size([256]),
        'encoder.layer.2.attention.output.LayerNorm.bias': torch.Size([256]),
        'encoder.layer.2.intermediate.dense.weight': torch.Size([1024, 256]),
        'encoder.layer.2.intermediate.dense.bias': torch.Size([1024]),
        'encoder.layer.2.output.dense.weight': torch.Size([256, 1024]),
        'encoder.layer.2.output.dense.bias': torch.Size([256]),
        'encoder.layer.2.output.LayerNorm.weight': torch.Size([256]),
        'encoder.layer.2.output.LayerNorm.bias': torch.Size([256]),
        'encoder.layer.3.attention.self.query.weight': torch.Size([256, 256]),
        'encoder.layer.3.attention.self.query.bias': torch.Size([256]),
        'encoder.layer.3.attention.self.key.weight': torch.Size([256, 256]),
        'encoder.layer.3.attention.self.key.bias': torch.Size([256]),
        'encoder.layer.3.attention.self.value.weight': torch.Size([256, 256]),
        'encoder.layer.3.attention.self.value.bias': torch.Size([256]),
        'encoder.layer.3.attention.output.dense.weight': torch.Size([256, 256]),
        'encoder.layer.3.attention.output.dense.bias': torch.Size([256]),
        'encoder.layer.3.attention.output.LayerNorm.weight': torch.Size([256]),
        'encoder.layer.3.attention.output.LayerNorm.bias': torch.Size([256]),
        'encoder.layer.3.intermediate.dense.weight': torch.Size([1024, 256]),
        'encoder.layer.3.intermediate.dense.bias': torch.Size([1024]),
        'encoder.layer.3.output.dense.weight': torch.Size([256, 1024]),
        'encoder.layer.3.output.dense.bias': torch.Size([256]),
        'encoder.layer.3.output.LayerNorm.weight': torch.Size([256]),
        'encoder.layer.3.output.LayerNorm.bias': torch.Size([256]),
    }
    state_dict = {k: torch.rand(v) for k, v in layout.items()}

    model = Bertish()

    try:
        model.load_transformers_bert_state_dict(state_dict)
    except RuntimeError as err:
        pytest.fail(f'failed to load bert state dict from transformers style model! raised {err}')
