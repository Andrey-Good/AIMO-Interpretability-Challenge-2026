# Учебные выдержки, не submission и не самостоятельный модуль.
# Оригинал: solutions/trained-probe/probe_inference.py
# Маршрут: ../route/02-probe.md; соответствие: source_map.json.
from __future__ import annotations

if __name__ == "__main__":
    raise SystemExit("Это выдержки для чтения. Запускай travel_guide/labs/02_probe.py")


def _layer_index(artifact: ProbeArtifact) -> int:
    """Resolve which model layer's hidden state the probe reads.

    For the ensemble format, prefer the strategy's ``layer_index`` and fall back
    to the globally selected ``best_layer_index``.
    """
    # GUIDE: это индекс в output.hidden_states, не произвольный номер hook.
    if artifact.kind == "pickle":
        strategy = artifact.data.get("recommended_strategy") or {}
        return int(strategy.get("layer_index", artifact.data["best_layer_index"]))
    return int(artifact.data["layer_index"])


def _encode_problem(problem_text: str, artifact: ProbeArtifact) -> Any:
    """Run one forward pass and return the probe's input hidden-state vector.

    The vector is the last prompt token's hidden state at the probe's layer,
    mirroring how the probes were trained. Returned as a float32 numpy array.
    """
    # GUIDE: эти помощники определены в оригинальном файле.
    # Загрузка находится ВНУТРИ обработки одной задачи — это не кеш модели.
    tokenizer, model = _load_model(artifact.model_id)
    prompt = _build_prompt(tokenizer, problem_text, artifact.system_prompt)
    model_inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    model_inputs = {name: value.to(model.device) for name, value in model_inputs.items()}

    # No gradients or KV cache needed; we only want the hidden states.
    # GUIDE: input_ids обычно [1,T]. Здесь forward, не generate.
    # eval() выставлен в _load_model; no_grad() отдельно отключает граф градиентов.
    with torch.no_grad():
        output = model(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    hidden_states = output.hidden_states
    if hidden_states is None:
        raise RuntimeError("model forward pass returned no hidden states")

    # hidden_states is a tuple of (num_layers + 1) tensors, each
    # (batch, seq_len, hidden_dim). Validate the layer index is in range.
    layer_index = _layer_index(artifact)
    if not -len(hidden_states) <= layer_index < len(hidden_states):
        raise RuntimeError(
            f"probe layer {layer_index} is outside model hidden-state range "
            f"0..{len(hidden_states) - 1}"
        )
    # [0, -1, :] => batch item 0, final token, full hidden vector.
    # GUIDE: [1,T,D] -> [D]. -1 относится к отформатированному промпту,
    # включая возможный служебный токен начала ответа, НЕ к generated text.
    # float() -> float32; cpu() -> CPU; numpy() -> массив для линейной пробы.
    return hidden_states[layer_index][0, -1, :].detach().float().cpu().numpy()


def mean_ensemble_margin(vector: Any, artifact: ProbeArtifact) -> float:
    """Average signed margin across every probe in the ensemble.

    The ensemble spans groups (e.g. cross-validation folds), each holding a
    stack of per-seed probes for the chosen layer. We compute
    ``weights @ vector + bias - threshold`` for all of them and return the mean;
    a non-negative mean is the "robust" decision.
    """
    layer_index = _layer_index(artifact)
    margins = []
    for group in artifact.data["groups"]:
        probes = group.get("probes", {})
        # Layer keys may be stored as ints or strings; accept either.
        probe = probes.get(layer_index, probes.get(str(layer_index)))
        if probe is None:
            continue

        # weights: (n_seeds, hidden_dim); normalize a 1-D probe to 2-D.
        weights = np.asarray(probe["weights"], dtype=np.float32)
        if weights.ndim == 1:
            weights = weights.reshape(1, -1)
        if weights.ndim != 2 or weights.shape[1] != vector.shape[0]:
            raise RuntimeError(
                f"probe dimension mismatch: got {vector.shape[0]}, expected {weights.shape[-1]}"
            )

        # bias/threshold are per-seed vectors aligned with weights' first axis.
        bias = np.asarray(probe["bias"], dtype=np.float32).reshape(-1)
        threshold = np.asarray(probe["threshold"], dtype=np.float32).reshape(-1)
        if bias.size != weights.shape[0] or threshold.size != weights.shape[0]:
            raise RuntimeError("probe ensemble arrays have inconsistent seed dimensions")

        # One margin per seed in this group; collect across all groups.
        # GUIDE: [S,D] @ [D] + [S] - [S] -> [S]. Это НЕ вероятности.
        # extend сохраняет каждую пробу, а не одно среднее на группу.
        margins.extend((weights @ vector + bias - threshold).astype(float).tolist())

    if not margins:
        raise RuntimeError(f"artifact contains no probes for layer {layer_index}")
    # float64 mean to avoid accumulation error across many seeds/folds.
    # GUIDE: отрицательное среднее -> False в _predict_problem.
    # Равенство нулю -> True. Голосования большинством здесь нет.
    return float(np.mean(np.asarray(margins, dtype=np.float64)))
