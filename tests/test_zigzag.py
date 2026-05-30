import torch
import pytest

from pixzig_field.zigma import ZigMaBackbone, get_zigzag_indices, invert_indices


def _assert_valid_paths(grid_h: int, grid_w: int) -> None:
    paths = get_zigzag_indices(grid_h, grid_w, mode="zigzagN16")
    expected = torch.arange(grid_h * grid_w)

    assert len(paths) == 16
    for path in paths:
        assert path.numel() == grid_h * grid_w
        assert torch.equal(torch.sort(path).values, expected)
        assert torch.equal(path[invert_indices(path)], expected)


def test_zigzag_square_grid():
    _assert_valid_paths(4, 4)


def test_zigzag_rectangular_grid():
    _assert_valid_paths(3, 5)


def test_zigzag_indices_cache_is_not_mutated_by_callers():
    original = get_zigzag_indices(2, 3, mode="zigzagN1")[0]
    original[0] = -1

    fresh = get_zigzag_indices(2, 3, mode="zigzagN1")[0]

    assert fresh[0] >= 0
    assert torch.equal(torch.sort(fresh).values, torch.arange(6))


def test_zigzag_indices_reject_invalid_arguments():
    with pytest.raises(ValueError, match="grid_h"):
        get_zigzag_indices(0, 2)

    with pytest.raises(TypeError, match="integer"):
        get_zigzag_indices(True, 2)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="unsupported scan mode"):
        get_zigzag_indices(2, 2, mode="rows")

    with pytest.raises(ValueError, match="positive"):
        get_zigzag_indices(2, 2, mode="zigzagN0")


def test_invert_indices_rejects_invalid_permutations():
    with pytest.raises(TypeError, match="torch.long"):
        invert_indices(torch.tensor([0, 1], dtype=torch.int32))

    with pytest.raises(ValueError, match="1D"):
        invert_indices(torch.zeros(1, 2, dtype=torch.long))

    with pytest.raises(ValueError, match="non-empty"):
        invert_indices(torch.empty(0, dtype=torch.long))

    with pytest.raises(ValueError, match=r"\[0, len"):
        invert_indices(torch.tensor([0, 2], dtype=torch.long))

    with pytest.raises(ValueError, match="duplicates"):
        invert_indices(torch.tensor([0, 0], dtype=torch.long))


def test_backbone_accepts_broadcast_conditioning():
    backbone = ZigMaBackbone(hidden_dim=32, depth=1, state_dim=8, scan_mode="zigzagN1", cond_dim=4)
    tokens = torch.randn(2, 4, 32)
    t_embed = torch.randn(1, 32)
    cond = torch.randn(1, 3, 4)

    output = backbone(tokens, t_embed, cond=cond, grid_size=(2, 2))

    assert output.shape == tokens.shape
    assert torch.isfinite(output).all()


def test_backbone_rejects_invalid_constructor_arguments():
    with pytest.raises(TypeError, match="integer"):
        ZigMaBackbone(hidden_dim=True, depth=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="depth"):
        ZigMaBackbone(hidden_dim=32, depth=0)

    with pytest.raises(ValueError, match="mamba_backend"):
        ZigMaBackbone(hidden_dim=32, depth=1, mamba_backend="bad")

    with pytest.raises(ValueError, match="unsupported scan mode"):
        ZigMaBackbone(hidden_dim=32, depth=1, scan_mode="bad")


def test_backbone_rejects_invalid_forward_inputs():
    backbone = ZigMaBackbone(hidden_dim=32, depth=1, state_dim=8, scan_mode="zigzagN1", cond_dim=4)
    tokens = torch.randn(2, 4, 32)
    t_embed = torch.randn(2, 32)

    with pytest.raises(TypeError, match="floating point"):
        backbone(torch.ones(2, 4, 32, dtype=torch.long), t_embed, grid_size=(2, 2))

    with pytest.raises(ValueError, match="expected tokens"):
        backbone(torch.randn(4, 32), t_embed, grid_size=(2, 2))

    with pytest.raises(ValueError, match="last dimension"):
        backbone(torch.randn(2, 4, 16), t_embed, grid_size=(2, 2))

    with pytest.raises(ValueError, match="t_embed batch size"):
        backbone(tokens, torch.randn(3, 32), grid_size=(2, 2))

    with pytest.raises(TypeError, match="grid_size"):
        backbone(tokens, t_embed, grid_size=[2, 2])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="expects 6 tokens"):
        backbone(tokens, t_embed, grid_size=(2, 3))

    with pytest.raises(ValueError, match="cond last dimension"):
        backbone(tokens, t_embed, cond=torch.randn(2, 5), grid_size=(2, 2))

    unconditioned = ZigMaBackbone(hidden_dim=32, depth=1, state_dim=8, scan_mode="zigzagN1")
    with pytest.raises(ValueError, match="cond_dim"):
        unconditioned(tokens, t_embed, cond=torch.randn(2, 4), grid_size=(2, 2))


def test_zigma_expand_changes_state_capacity():
    narrow = ZigMaBackbone(hidden_dim=32, depth=1, state_dim=8, expand=1)
    expanded = ZigMaBackbone(hidden_dim=32, depth=1, state_dim=8, expand=2)

    assert narrow.blocks[0].state_dim == 8
    assert expanded.blocks[0].state_dim == 16
    assert sum(parameter.numel() for parameter in expanded.parameters()) > sum(
        parameter.numel() for parameter in narrow.parameters()
    )
