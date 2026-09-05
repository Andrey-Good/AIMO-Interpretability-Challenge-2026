"""Вероятности -> gather -> маска -> настоящие 14 признаков.

Распределения выдуманы, модель не загружается. Для наглядности храним крошечный
[B,G,V]; реальный online collector НЕ сохраняет весь такой массив.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from _repo import load_module, read_literal

PREFIX = "solutions/uncertainty-profiling/uncertainty_profile/"


def main() -> None:
    metrics = load_module(PREFIX + "metrics.py", "guide_metrics")
    feature_names = read_literal(PREFIX + "config.py", "FEATURE_NAMES")
    probabilities = torch.tensor([
        [[0.7, 0.1, 0.1, 0.1], [0.2, 0.4, 0.3, 0.1], [0.1, 0.1, 0.1, 0.7]],
        [[0.1, 0.6, 0.2, 0.1], [0.1, 0.1, 0.1, 0.7], [0.1, 0.1, 0.1, 0.7]],
    ])
    logits = probabilities.log()
    log_probs = torch.log_softmax(logits, dim=-1)
    generated_ids = logits.argmax(dim=-1)
    selected = log_probs.gather(-1, generated_ids.unsqueeze(-1)).squeeze(-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    top2_probs, top2_ids = log_probs.exp().topk(k=2, dim=-1)
    # Условный токен 3 означает EOS и используется как pad.
    valid_mask = generated_ids != 3
    print("logits [B,G,V]:", tuple(logits.shape))
    print("generated IDs [B,G]:", generated_ids.tolist())
    print("selected log-probs [B,G]:", selected.tolist())
    print("valid mask:", valid_mask.tolist())
    assert valid_mask.sum(dim=1).tolist() == [2, 1]

    rows = []
    for index in range(2):
        mask = valid_mask[index]
        row = metrics.compute_generation_confidence_metrics(
            log_probs=selected[index][mask].numpy(),
            probs=selected[index][mask].exp().numpy(),
            entropy=entropy[index][mask].numpy(),
            top1_probs=top2_probs[index, :, 0][mask].numpy(),
            top2_margins=(top2_probs[index, :, 0] - top2_probs[index, :, 1])[mask].numpy(),
            selected_is_top1=(generated_ids[index] == top2_ids[index, :, 0])[mask].numpy(),
            min_k_fraction=0.2, high_conf_threshold=0.9, low_conf_threshold=0.1,
        )
        rows.append(row)
        print(f"row {index}: tokens={row['generation_num_tokens']}, "
              f"NLL={row['generation_mean_nll']:.6f}, PPL={row['generation_ppl']:.6f}")

    # То же упорядочивание, что DataFrame(columns=artifact.feature_names).
    features = np.asarray([[row[name] for name in feature_names] for row in rows])
    assert features.shape == (2, 14)
    assert "generation_num_tokens" not in feature_names
    assert np.isclose(rows[0]["generation_ppl"], 1 / math.sqrt(0.7 * 0.4), atol=1e-6)
    assert np.isclose(rows[1]["generation_ppl"], 1 / 0.6, atol=1e-6)
    assert np.isclose(rows[0]["generation_min_k_logprob"], math.log(0.4), atol=1e-6)
    assert all(row["generation_frac_selected_is_top1"] == 1.0 for row in rows)
    uniform = torch.full((4,), 0.25)
    assert math.isclose(float(-(uniform * uniform.log()).sum()), math.log(4), rel_tol=1e-6)
    print("features:", features.shape, "; uniform entropy = ln(4)")
    print("OK. Применена настоящая metrics-функция; это не оценка реальной LLM.")


if __name__ == "__main__":
    main()
