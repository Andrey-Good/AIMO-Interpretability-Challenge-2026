"""Что наблюдаем: последний токен промпта на выбранных слоях, без генерации.
B — задачи в батче; T — длина с padding; K — слои; D — ширина состояния.
"""
from collections.abc import Sequence
import torch
from torch import Tensor, nn


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
