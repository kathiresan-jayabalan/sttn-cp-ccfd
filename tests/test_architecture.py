import torch
import pytest

from src.architecture import STTNCP, SpatialTransformerBlock, infonce_loss


def test_spatial_attention_uses_original_feature_count():
    layer = SpatialTransformerBlock(input_dim=5, embed_dim=8, n_heads=2, mlp_dim=16)
    seen = {}

    def hook(module, inputs, output):
        seen["tokens"] = inputs[0].shape[1]

    handle = layer.transformer.register_forward_hook(hook)
    try:
        x = torch.randn(2, 3, 5)
        y = layer(x)
    finally:
        handle.remove()

    assert seen["tokens"] == 5
    assert y.shape == (2, 3, 8)


def test_model_classifies_complete_windows():
    model = STTNCP(input_dim=5, seq_len=4, embed_dim=8, n_blocks=1, n_heads=2, mlp_dim=16)
    x = torch.randn(3, 4, 5)
    logits = model(x)
    assert logits.shape == (3, 2)

    with pytest.raises(ValueError, match=r"shape \(B, T, D\)"):
        model(torch.randn(3, 5))


def test_infonce_requires_multiple_samples():
    with pytest.raises(ValueError, match="at least two"):
        infonce_loss(torch.randn(1, 4), torch.randn(1, 4))
