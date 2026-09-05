"""[1,T,D] -> [D] -> margins [S] -> bool на искусственных данных.

Нужны только NumPy и PyTorch. Настоящий scorer читается из исходника,
но Transformers, модель и pickle не загружаются. Никакого обучения здесь нет.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from _repo import load_functions


def main() -> None:
    # Три отдельных состояния: вход и два условных слоя. Это НЕ выход LLM.
    hidden_states = tuple(torch.zeros(1, 4, 3) for _ in range(3))
    hidden_states[2][0, -1, :] = torch.tensor([1.0, 2.0, -1.0])
    h = hidden_states[2][0, -1, :]
    print("tuple length:", len(hidden_states))
    print("H_2:", tuple(hidden_states[2].shape), "-> h:", tuple(h.shape), h.tolist())
    assert h.shape == (3,)

    weights = torch.tensor([[0.5, -1.0, 2.0], [-1.0, 1.0, -0.5]])
    bias = torch.tensor([0.2, 1.0])
    threshold = torch.tensor([0.3, 0.1])
    margins = weights @ h + bias - threshold
    torch.testing.assert_close(margins, torch.tensor([-3.6, 2.4]))
    print("W:", tuple(weights.shape), "-> margins:", margins.tolist())
    print("mean margin:", margins.mean().item(), "->", bool(margins.mean() >= 0))

    # Сравниваем нашу формулу с функцией оригинального baseline.
    functions = load_functions(
        "solutions/trained-probe/probe_inference.py",
        ["_layer_index", "mean_ensemble_margin"], {"np": np},
    )
    artifact = SimpleNamespace(kind="pickle", data={
        "best_layer_index": 2,
        "groups": [{"probes": {2: {"weights": weights.numpy(), "bias": bias.numpy(), "threshold": threshold.numpy()}}}],
    })
    actual = functions["mean_ensemble_margin"](h.numpy(), artifact)
    assert np.isclose(actual, -0.6, atol=1e-6)
    assert np.isclose(actual, margins.mean().item(), atol=1e-6)
    assert type(bool(actual >= 0)) is bool

    votes = torch.tensor([1.0, 1.0, -5.0])
    assert bool((votes >= 0).float().mean() > 0.5) is True
    assert bool(votes.mean() >= 0) is False
    print("[1,1,-5]: большинство за True, средний margin даёт False.")
    print("OK. PyTorch и исходная NumPy-функция совпали.")


if __name__ == "__main__":
    main()
