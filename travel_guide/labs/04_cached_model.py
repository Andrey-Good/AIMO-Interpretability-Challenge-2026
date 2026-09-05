"""Необязательный реальный forward. Только уже закешированная модель.

Нужны Transformers, PyTorch, локальные веса и достаточная память.
Это осмотр форм, НЕ точное воспроизведение промпта обученного probe.
Никаких автоматических скачиваний, обучения или загрузки pickle.
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, help="Точный HF checkpoint или локальный путь, не alias")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--problem", default="What is 2 + 2?")
    args = parser.parse_args()
    if args.max_tokens < 1 or not args.problem:
        parser.error("--max-tokens должен быть положительным, --problem — непустым")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Нужно окружение проекта с torch и transformers; ничего автоматически не устанавливается.") from exc
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA недоступна. Этот скрипт не переключает устройство молча.")
    dtype = torch.float32
    if args.device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading local checkpoint {args.model_id} on {args.device}, {dtype}.", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, dtype=dtype, local_files_only=True, trust_remote_code=False,
        ).to(args.device).eval()
        inputs = tokenizer(args.problem, return_tensors="pt", truncation=True, max_length=args.max_tokens)
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        states = output.hidden_states
        if states is None or not -len(states) <= args.layer < len(states):
            raise RuntimeError("Модель не вернула hidden_states либо выбран недопустимый индекс слоя.")
        print("input_ids:", tuple(inputs["input_ids"].shape))
        for index, state in enumerate(states):
            print(f"hidden_states[{index}]: {tuple(state.shape)}, {state.dtype}, {state.device}")
        vector = states[args.layer][0, -1, :].float().cpu()
        print("selected vector:", tuple(vector.shape), "; first five coordinates:", vector[:5].tolist())
        print("Этот промпт учебный: совместимость с артефактом probe не проверялась.")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Forward не выполнен: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
