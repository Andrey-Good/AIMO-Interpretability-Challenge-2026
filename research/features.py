"""Промпт → LLM → последний немаскированный токен выбранных слоёв.

B — задачи; T — токены; K — выбранные состояния; D — ширина состояния.
Tokenizer/template/truncation должны быть явно записаны в feature_spec извне.
Это состояния ПРОМПТА, не сгенерированного решения. Метки сюда не передаются.
"""
from collections.abc import Mapping, Sequence
import torch
from torch import Tensor, nn


def extract_features(
    llm: nn.Module, batch: Mapping[str, Tensor], layers: Sequence[int]
) -> Tensor:
    """Return detached float32 [B,K,D], preserving batch and layer order.

    Caller supplies tokenized, device-placed inputs. Works with left/right padding.
    Requested indices refer to output.hidden_states, NOT necessarily block numbers.
    Only standard input_ids/attention_mask are accepted to keep the contract explicit.
    """
    if set(batch) != {"input_ids", "attention_mask"}:
        raise ValueError("batch must contain exactly input_ids and attention_mask")
    ids, mask = batch["input_ids"], batch["attention_mask"]
    if ids.ndim != 2 or mask.shape != ids.shape or 0 in ids.shape:
        raise ValueError("nonempty input_ids and mask must have the same [B,T] shape")
    if ids.dtype != torch.long or mask.device != ids.device:
        raise ValueError("input_ids must be int64; mask must share its device")
    if not torch.all((mask == 0) | (mask == 1)) or not torch.all(mask.bool().any(dim=1)):
        raise ValueError("mask must be binary with at least one valid token per row")
    if not layers or any(type(i) is not int for i in layers):
        raise ValueError("layers must be a nonempty sequence of integer indices")
    if len(set(layers)) != len(layers):
        raise ValueError("duplicate layers would silently reweight features")

    # Индекс ПОСЛЕДНЕЙ единицы; sum(mask)-1 неверен при левом padding.
    positions = torch.arange(ids.shape[1], device=ids.device)
    last = positions.expand_as(ids).masked_fill(~mask.bool(), -1).max(dim=1).values
    previous_mode = llm.training
    llm.eval()
    try:
        with torch.no_grad():
            output = llm(**batch, output_hidden_states=True, return_dict=True, use_cache=False)
            states = output.hidden_states  # tuple: по [B,T,D] на каждый индекс
            if states is None or any(i < 0 or i >= len(states) for i in layers):
                raise ValueError("requested layer is outside hidden_states")
            chosen = []
            for i in layers:
                h = states[i]
                if h.ndim != 3 or h.shape[:2] != ids.shape:
                    raise ValueError("hidden state must have shape [B,T,D]")
                rows = torch.arange(ids.shape[0], device=h.device)
                chosen.append(h[rows, last.to(h.device), :].detach().float())
            # [B,D] для каждого слоя → [B,K,D]. Multi-device states copy to first.
            x = torch.stack([h.to(chosen[0].device) for h in chosen], dim=1)
            if not torch.isfinite(x).all():
                raise ValueError("non-finite features")
            return x
    finally:
        llm.train(previous_mode)
