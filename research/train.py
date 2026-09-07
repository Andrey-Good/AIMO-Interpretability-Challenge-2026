"""Прозрачное обучение ТОЛЬКО головы на уже извлечённых TRAIN-признаках."""
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def fit_predictor(
    predictor: nn.Module, features: Tensor, labels: Tensor, *,
    epochs: int = 10, batch_size: int = 32, learning_rate: float = 1e-3, seed: int = 42,
) -> list[float]:
    """X=[N,K,D], y=[N] с 0/1. Вернуть средний train loss каждой эпохи.

    Split, подбор параметров и validation находятся ВНЕ этой функции.
    Не передавайте сюда holdout. Seed задаёт порядок батчей; инициализацию головы
    надо отдельно зафиксировать перед build_predictor и записать в карточку.
    """
    if features.ndim != 3 or features.shape[0] < 1 or labels.shape != (features.shape[0],):
        raise ValueError("expected nonempty features [N,K,D] and labels [N]")
    if not torch.isfinite(features).all() or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("features must be finite and labels binary")
    if any(type(n) is not int or n < 1 for n in (epochs, batch_size)):
        raise ValueError("epochs and batch_size must be positive integers")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    parameter = next(predictor.parameters())
    x = features.detach().to(device=parameter.device, dtype=parameter.dtype)
    y = labels.detach().to(device=parameter.device, dtype=parameter.dtype)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history = []
    previous_mode = predictor.training
    predictor.train()
    try:
        for _ in range(epochs):
            indices = torch.randperm(len(y), generator=generator)
            total_loss = 0.0
            for start in range(0, len(y), batch_size):
                take = indices[start:start + batch_size].to(x.device)
                logits = predictor(x[take]).squeeze(-1)
                if logits.shape != y[take].shape:
                    raise ValueError("predictor must return [batch,1]")
                loss = F.binary_cross_entropy_with_logits(logits, y[take])
                if not torch.isfinite(loss):
                    raise ValueError("non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(take)
            history.append(total_loss / len(y))
    finally:
        predictor.train(previous_mode)
    return history
