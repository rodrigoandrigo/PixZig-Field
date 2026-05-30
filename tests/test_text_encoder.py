import torch
import pytest

from pixzig_field.text_encoder import QwenTextEncoder, masked_mean


def test_masked_mean_ignores_padding_tokens():
    tokens = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]],
            [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

    pooled = masked_mean(tokens, mask)

    assert torch.allclose(pooled[0], torch.tensor([2.0, 3.0]))
    assert torch.allclose(pooled[1], torch.tensor([2.0, 4.0]))


def test_masked_mean_rejects_invalid_shapes():
    tokens = torch.randn(2, 3, 4)

    with pytest.raises(TypeError, match="floating point"):
        masked_mean(torch.ones(2, 3, 4, dtype=torch.long))

    with pytest.raises(ValueError, match="shape"):
        masked_mean(torch.randn(2, 4))

    with pytest.raises(ValueError, match="attention_mask"):
        masked_mean(tokens, torch.ones(2, 2))


def test_qwen_wrapper_projects_precomputed_embeddings_with_pooling():
    encoder = QwenTextEncoder(output_dim=4, pooling="mean")
    embeddings = torch.randn(2, 3, 6)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    pooled = encoder(embeddings=embeddings, attention_mask=mask)

    assert pooled.shape == (2, 4)
    assert torch.isfinite(pooled).all()
    assert pooled.dtype == torch.float32


def test_qwen_wrapper_projects_bfloat16_embeddings_to_projection_dtype():
    encoder = QwenTextEncoder(output_dim=4, pooling="mean")
    embeddings = torch.randn(2, 3, 6, dtype=torch.bfloat16)

    pooled = encoder(embeddings=embeddings)

    assert pooled.shape == (2, 4)
    assert pooled.dtype == torch.float32


def test_qwen_projection_is_independent_of_global_seed():
    embeddings = torch.randn(2, 3, 6)

    torch.manual_seed(1)
    left = QwenTextEncoder(output_dim=4, pooling="mean")(embeddings=embeddings)
    torch.manual_seed(999)
    right = QwenTextEncoder(output_dim=4, pooling="mean")(embeddings=embeddings)

    assert torch.allclose(left, right)


def test_qwen_wrapper_returns_zero_conditioning_for_empty_text():
    encoder = QwenTextEncoder(output_dim=4, pooling="mean", cache_embeddings=True)

    cond = encoder([""])
    cached = encoder([""])

    assert cond.shape == (1, 4)
    assert cond.dtype == torch.float32
    assert torch.all(cond == 0)
    assert torch.allclose(cond, cached)


def test_qwen_wrapper_cached_conditioning_follows_encoder_dtype():
    encoder = QwenTextEncoder(output_dim=4, pooling="mean", cache_embeddings=True)

    encoder([""])
    encoder.to(dtype=torch.float16)
    cached = encoder([""])

    assert cached.dtype == torch.float16


def test_qwen_wrapper_can_keep_token_embeddings():
    encoder = QwenTextEncoder(output_dim=4, pooling="none")
    embeddings = torch.randn(2, 3, 6)

    tokens = encoder(embeddings=embeddings)

    assert tokens.shape == (2, 3, 4)


def test_qwen_wrapper_preserves_precomputed_dtype_without_projection():
    encoder = QwenTextEncoder(output_dim=None, pooling="mean")
    embeddings = torch.randn(2, 3, 6, dtype=torch.float64)

    pooled = encoder(embeddings=embeddings)

    assert pooled.dtype == torch.float64
    assert pooled.device == embeddings.device


def test_qwen_wrapper_cache_is_not_mutable_by_callers():
    encoder = QwenTextEncoder(output_dim=4, pooling="mean", cache_embeddings=True)

    first = encoder([""])
    first.fill_(123.0)
    second = encoder([""])

    assert torch.all(second == 0)


def test_qwen_wrapper_rejects_invalid_constructor_arguments():
    with pytest.raises(ValueError, match="model_name"):
        QwenTextEncoder(model_name="")

    with pytest.raises(TypeError, match="output_dim"):
        QwenTextEncoder(output_dim=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="max_length"):
        QwenTextEncoder(max_length=0)

    with pytest.raises(TypeError, match="cache_embeddings"):
        QwenTextEncoder(cache_embeddings="yes")  # type: ignore[arg-type]


def test_qwen_wrapper_rejects_invalid_forward_arguments():
    encoder = QwenTextEncoder(output_dim=4, pooling="mean")

    with pytest.raises(ValueError, match="either text or precomputed embeddings"):
        encoder(["caption"], embeddings=torch.randn(1, 2, 6))

    with pytest.raises(TypeError, match="list or tuple"):
        encoder("caption")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="at least one prompt"):
        encoder([])

    with pytest.raises(TypeError, match="strings"):
        encoder(["ok", 1])  # type: ignore[list-item]

    with pytest.raises(TypeError, match="floating point"):
        encoder(embeddings=torch.ones(1, 2, 6, dtype=torch.long))

    with pytest.raises(ValueError, match="attention_mask"):
        encoder(embeddings=torch.randn(1, 6), attention_mask=torch.ones(1, 2))

    token_encoder = QwenTextEncoder(output_dim=4, pooling="none")
    with pytest.raises(ValueError, match="pooling='none'"):
        token_encoder(embeddings=torch.randn(1, 6))
