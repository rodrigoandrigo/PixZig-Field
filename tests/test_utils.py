import pytest
import torch

from pixzig_field.utils import count_parameters, format_duration


class TinyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trainable = torch.nn.Parameter(torch.zeros(2, 3))
        self.frozen = torch.nn.Parameter(torch.zeros(5), requires_grad=False)


def test_count_parameters_counts_only_trainable_parameters():
    model = TinyModule()

    assert count_parameters(model) == 6


def test_count_parameters_rejects_non_module():
    with pytest.raises(TypeError, match="nn.Module"):
        count_parameters(object())  # type: ignore[arg-type]


def test_format_duration_formats_common_ranges():
    assert format_duration(-1.0) == "0s"
    assert format_duration(9.9) == "9s"
    assert format_duration(65.0) == "1m05s"
    assert format_duration(3661.0) == "1h01m01s"


def test_format_duration_rejects_invalid_values():
    with pytest.raises(TypeError, match="real number"):
        format_duration(True)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="real number"):
        format_duration("1")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="finite"):
        format_duration(float("nan"))

    with pytest.raises(ValueError, match="finite"):
        format_duration(float("inf"))
