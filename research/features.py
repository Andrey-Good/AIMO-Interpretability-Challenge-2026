"""Что наблюдаем: последний токен промпта на выбранных слоях, без генерации.
B — задачи в батче; T — длина с padding; K — слои; D — ширина состояния.
"""
from collections.abc import Sequence
import torch
from torch import Tensor, nn
import hashlib


def stable_positions(generation_positions: Sequence[int], *, seed: int, run_id: str) -> list[tuple[int, str]]:
    """Contract selection: first 8, stable-hash middle 16, last 8; never pad."""
    positions = sorted(set(generation_positions))
    if not positions:
        return []
    first, last = positions[:8], positions[-8:]
    middle = sorted(positions, key=lambda p: hashlib.sha256(f"v1:{seed}:{run_id}:{p}".encode()).digest())[:16]
    roles = {}
    for p in first: roles[p] = "first8"
    for p in middle: roles[p] = roles.get(p, "") + "+hash16"
    for p in last: roles[p] = roles.get(p, "") + "+last8"
    return [(p, roles[p].strip("+")) for p in sorted(roles)]


def output_statistics(logits: Tensor, selected_id: int, *, greedy: bool) -> Tensor:
    """Top-20 raw logits plus logZ, entropy, raw/policy selected log-probability."""
    if logits.ndim != 1 or not 0 <= selected_id < logits.numel() or not torch.isfinite(logits).all():
        raise ValueError("finite one-dimensional logits and valid selected_id required")
    z = logits.float()
    logz = torch.logsumexp(z, 0)
    logp = z - logz
    entropy = -(logp.exp() * logp).sum()
    values, ids = torch.topk(z, min(20, z.numel()))
    policy = torch.zeros((), dtype=torch.float32, device=z.device) if greedy else logp[selected_id]
    return torch.cat((ids.to(torch.float32), values, torch.stack((logz, entropy, logp[selected_id], policy))))


def segment_generation(ids: Sequence[int], close_marker_ids: Sequence[int], *, eos_id: int | None = None) -> tuple[list[str], str]:
    """Q is handled from template offsets; this strictly labels generated R/A or unknown."""
    if not close_marker_ids:
        raise ValueError("actual close-marker token IDs are required")
    ids, marker = list(ids), list(close_marker_ids)
    hits = [i for i in range(len(ids)-len(marker)+1) if ids[i:i+len(marker)] == marker]
    service = {i for start in hits for i in range(start, start + len(marker))}
    service.update(i for i, token in enumerate(ids) if token == eos_id)
    if len(hits) != 1:
        return (["service" if i in service else "unknown" for i in range(len(ids))], "absent" if not hits else "ambiguous")
    close = hits[0]
    roles = []
    for i, token in enumerate(ids):
        if i in service:
            roles.append("service")
        else:
            roles.append("R" if i < close else "A")
    return roles, "confirmed"


@torch.no_grad()
def extract_features(llm: nn.Module, tokenizer, texts: Sequence[str], *,
                     layers: Sequence[int], max_length: int, template: str) -> Tensor:
    """Строки → токены [B,T] → состояния [B,T,D] → признаки [B,K,D] на CPU.
    Индексы layers относятся к tuple hidden_states, не автоматически к номерам блоков.
    tokenizer/model загружены обвязкой. Меток y здесь нет.
    """
    if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("Expected nonempty problem strings")
    if not layers or any(type(i) is not int or i < 0 for i in layers) or len(set(layers)) != len(layers):
        raise ValueError("Use distinct nonnegative hidden-state indices")
    if type(max_length) is not int or max_length < 1 or template not in {"plain", "chat"}:
        raise ValueError("Invalid tokenization settings")
    prompts = list(texts)
    if template == "chat":
        # Нет молчаливого fallback: другой шаблон означает другой эксперимент.
        prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True
        ) for t in texts]
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    encoded = tokenizer(prompts, padding=True, truncation=True, max_length=max_length,
                        add_special_tokens=template == "plain", return_tensors="pt")
    device = llm.get_input_embeddings().weight.device
    batch = {key: encoded[key].to(device) for key in ("input_ids", "attention_mask")}
    mask = batch["attention_mask"]
    if mask.ndim != 2 or mask.shape != batch["input_ids"].shape or mask.shape[1] == 0:
        raise ValueError("Expected input_ids and mask with matching nonempty [B,T]")
    if not torch.all((mask == 0) | (mask == 1)) or not mask.bool().any(dim=1).all():
        raise ValueError("Every example needs a binary mask with a real token")
    # Последняя ЕДИНИЦА маски; работает и при левом, и при правом padding.
    last = torch.arange(mask.shape[1], device=device).expand_as(mask)
    last = last.masked_fill(~mask.bool(), -1).amax(dim=1)
    llm.eval()
    states = llm(**batch, output_hidden_states=True, return_dict=True,
                 use_cache=False).hidden_states
    if states is None or max(layers) >= len(states):
        raise ValueError("Requested state index does not exist")
    selected = []
    for layer in layers:
        h = states[layer]  # [B,T,D]
        if h.ndim != 3 or h.shape[:2] != mask.shape:
            raise ValueError("Unexpected hidden-state shape")
        rows = torch.arange(len(texts), device=h.device)
        selected.append(h[rows, last.to(h.device)].float().cpu())  # [B,D]
    x = torch.stack(selected, dim=1)  # [B,K,D], layer order preserved
    if not torch.isfinite(x).all():
        raise ValueError("Nonfinite features")
    return x
