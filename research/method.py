"""Весь научный путь на одном экране. Обвязка передаёт уже загруженные объекты."""
from collections.abc import Mapping, Sequence
import math
import torch
from torch import Tensor, nn
from research.features import extract_features


def predict(
    llm: nn.Module,
    predictor: nn.Module,
    batch: Mapping[str, Tensor],
    layers: Sequence[int],
    threshold: float = 0.5,
) -> list[bool]:
    """Токены → состояния [B,K,D] → logit [B] → вероятность → bool.

    Этот predict работает с непустым БАТЧЕМ. solution adapter позже отвечает
    за [] и разбиение списка строк. Он должен импортировать эти же функции.
    """
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and in [0,1]")
    x = extract_features(llm, batch, layers)
    # Перенос к голове — транспорт, не изменение научных признаков.
    parameter = next(predictor.parameters())
    x = x.to(device=parameter.device, dtype=parameter.dtype)
    previous_mode = predictor.training
    predictor.eval()
    try:
        with torch.no_grad():
            logits = predictor(x).squeeze(-1)
            if logits.shape != (x.shape[0],) or not torch.isfinite(logits).all():
                raise ValueError("predictor must return one finite logit per example")
            probabilities = torch.sigmoid(logits)
            return (probabilities >= threshold).cpu().tolist()
    finally:
        predictor.train(previous_mode)
