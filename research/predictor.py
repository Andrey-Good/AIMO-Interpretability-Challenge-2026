"""Архитектура головы: только преобразование [B,K,D] → [B,1]."""
from torch import nn


def build_predictor(hidden_dim: int, layer_count: int, width: int = 0) -> nn.Module:
    """width=0: линейная проба; width>0: одна скрытая ReLU-прослойка.

    Последний выход — logit (не вероятность). Sigmoid применяется при прогнозе,
    а при обучении включён в binary_cross_entropy_with_logits.
    """
    if any(type(n) is not int for n in (hidden_dim, layer_count, width)):
        raise ValueError("dimensions must be integers")
    if min(hidden_dim, layer_count) < 1 or width < 0:
        raise ValueError("invalid predictor dimensions")
    input_dim = hidden_dim * layer_count
    if width == 0:
        return nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(input_dim, 1))
    return nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(input_dim, width),
        nn.ReLU(),
        nn.Linear(width, 1),
    )
