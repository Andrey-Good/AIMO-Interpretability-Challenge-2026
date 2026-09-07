"""Голова и применение: видимая архитектура, без загрузки файлов."""
import torch
from torch import Tensor, nn


def build_predictor(input_dim: int, width: int = 0) -> nn.Module:
    """[B,K*D] → [B,1]. width=0: линейная проба; иначе один скрытый слой."""
    if type(input_dim) is not int or input_dim < 1 or type(width) is not int or width < 0:
        raise ValueError("Invalid predictor dimensions")
    if width == 0:
        return nn.Linear(input_dim, 1)
    return nn.Sequential(nn.Linear(input_dim, width), nn.ReLU(), nn.Linear(width, 1))


def normalize(x: Tensor, mean: Tensor, scale: Tensor) -> Tensor:
    """[B,K,D] → [B,K*D]; mean/scale получены ТОЛЬКО из train."""
    return (x.flatten(start_dim=1) - mean) / scale


@torch.no_grad()
def predict(predictor: nn.Module, x: Tensor, mean: Tensor, scale: Tensor) -> Tensor:
    """Состояния → обучающая нормировка → голова → вероятность [B]."""
    predictor.eval()
    logits = predictor(normalize(x, mean, scale)).squeeze(-1)
    if logits.shape != (len(x),) or not torch.isfinite(logits).all():
        raise ValueError("Expected one finite logit per problem")
    return torch.sigmoid(logits)
