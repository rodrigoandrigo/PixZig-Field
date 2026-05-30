import torch
import pytest

from pixzig_field.ema import EMAModel


def test_ema_respects_update_interval_and_warmup():
    model = torch.nn.Linear(2, 2)
    ema = EMAModel(model, decay=0.9, update_every=2, warmup_steps=2, min_decay=0.0)
    before = {name: value.clone() for name, value in ema.module.state_dict().items()}

    with torch.no_grad():
        model.weight.add_(1.0)

    assert ema.update(model) is False
    assert ema.num_update_calls == 1
    assert ema.num_updates == 0
    for name, value in ema.module.state_dict().items():
        assert torch.allclose(value, before[name])

    assert ema.update(model) is True
    assert ema.num_update_calls == 2
    assert ema.num_updates == 1

    expected_decay = 0.9 * 0.5
    expected_weight = before["weight"] * expected_decay + model.weight.detach() * (1.0 - expected_decay)
    assert torch.allclose(ema.module.weight, expected_weight)


def test_ema_copy_to_model():
    model = torch.nn.Linear(2, 2)
    target = torch.nn.Linear(2, 2)
    ema = EMAModel(model)

    ema.copy_to(target)

    for left, right in zip(ema.module.parameters(), target.parameters()):
        assert torch.allclose(left, right)


def test_ema_updates_frozen_parameters_and_buffers():
    model = torch.nn.BatchNorm1d(2)
    ema = EMAModel(model, decay=0.5)
    model.weight.requires_grad_(False)

    with torch.no_grad():
        model.weight.add_(2.0)
        model.running_mean.add_(3.0)

    assert ema.update(model) is True

    assert torch.allclose(ema.module.weight, torch.ones_like(model.weight) * 2.0)
    assert torch.allclose(ema.module.running_mean, model.running_mean)


def test_ema_rejects_invalid_hyperparameters_and_state():
    model = torch.nn.Linear(2, 2)

    with pytest.raises(ValueError, match="decay"):
        EMAModel(model, decay=float("nan"))
    with pytest.raises(ValueError, match="update_every"):
        EMAModel(model, update_every=0)
    with pytest.raises(ValueError, match="warmup_steps"):
        EMAModel(model, warmup_steps=-1)
    with pytest.raises(ValueError, match="min_decay"):
        EMAModel(model, decay=0.5, min_decay=0.6)

    ema = EMAModel(model)
    state = ema.state_dict()
    state["num_updates"] = -1
    with pytest.raises(ValueError, match="non-negative"):
        ema.load_state_dict(state)
