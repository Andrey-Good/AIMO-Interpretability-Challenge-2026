"""One-input-at-a-time Qwen activation collection.

This is plumbing for the visible collection method in ``experiment.py``.  It
deliberately keeps only one forward pass worth of hook tensors on the GPU.
"""
from __future__ import annotations

from contextlib import contextmanager
from collections import deque
import hashlib
import json
from pathlib import Path
import time

import torch

from research._activation_store import Welford, bf16_bits, geometry_channels, write_completed
from research.features import segment_generation, stable_positions


LAYERS = tuple(range(2, 36, 3))
RAW_FAMILY_COUNT = 12
FORMAT_VERSION = 1


def _hash(*parts: object) -> bytes:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).digest()


def raw_families(cases: list[dict]) -> set[str]:
    """Fixed before collection: never select raw examples from outcomes."""
    return {f for f in sorted({c["family_id"] for c in cases}, key=lambda f: _hash("raw-v1", f))[:RAW_FAMILY_COUNT]}


def _content_positions(tokenizer, ids: list[int], roles: list[str]) -> list[int]:
    """Unknown phase remains content, but marker/EOS and whitespace-only pieces do not."""
    return [i for i, (token, role) in enumerate(zip(ids, roles))
            if role != "service" and tokenizer.decode([token], skip_special_tokens=False).strip()]


def raw_prompt_positions(question: list[int]) -> list[int]:
    """The raw subset has 32 evenly spaced question tokens."""
    if not question:
        return []
    if len(question) <= 32:
        return list(question)
    return sorted(set(question[round(i * (len(question) - 1) / 31)] for i in range(32)))


def raw_generated_positions(generation: list[int]) -> list[int]:
    """The raw subset's generated positions: first/hash/last 32, deduplicated."""
    if not generation:
        return []
    hashed = sorted(generation, key=lambda p: _hash("raw-position-v1", p))[:32]
    return sorted(set(generation[:32] + hashed + generation[-32:]))


class StreamingSelector:
    """Keep only vectors that can still enter a final sparse/raw archive."""
    def __init__(self, *, seed: int, run_id: str, raw: bool):
        self.seed, self.run_id, self.raw = seed, run_id, raw
        self.s_first, self.s_hash, self.s_last = [], [], deque(maxlen=8)
        self.r_first, self.r_hash, self.r_last = [], [], deque(maxlen=32)
        self.count = 0; self.peak_vector_frames = 0

    @staticmethod
    def _keep(items, item, size, key):
        items.append(item)
        items.sort(key=key)
        del items[size:]

    def add(self, frame: dict) -> None:
        p = frame["position"]
        if self.count < 8: self.s_first.append(frame)
        self._keep(self.s_hash, frame, 16, lambda f: hashlib.sha256(f"v1:{self.seed}:{self.run_id}:{f['position']}".encode()).digest())
        self.s_last.append(frame)
        if self.raw:
            if self.count < 32: self.r_first.append(frame)
            self._keep(self.r_hash, frame, 32, lambda f: _hash("raw-position-v1", f["position"]))
            self.r_last.append(frame)
        self.count += 1
        self.peak_vector_frames = max(self.peak_vector_frames, self.vector_frames_held)

    @staticmethod
    def _union(*groups):
        return {f["position"]: f for group in groups for f in group}

    def sparse(self) -> list[tuple[dict, int]]:
        chosen = self._union(self.s_first, self.s_hash, self.s_last)
        result = []
        for p, frame in sorted(chosen.items()):
            role = (1 if any(x["position"] == p for x in self.s_first) else 0) | (2 if any(x["position"] == p for x in self.s_hash) else 0) | (4 if any(x["position"] == p for x in self.s_last) else 0)
            result.append((frame, role))
        return result

    def raw_frames(self) -> list[dict]:
        return [f for _, f in sorted(self._union(self.r_first, self.r_hash, self.r_last).items())]

    @property
    def vector_frames_held(self) -> int:
        return len(self._union(self.s_first, self.s_hash, self.s_last, self.r_first, self.r_hash, self.r_last))


class HookCapture:
    """Capture actual Qwen residual locations without reconstructing them."""
    def __init__(self, model):
        base = getattr(model, "model", None)
        blocks = getattr(base, "layers", None)
        if blocks is None:
            raise ValueError("collector requires a Qwen-style model.model.layers backbone")
        self.blocks = list(blocks)
        self.handles = []
        self.values: dict[int, dict[str, torch.Tensor]] = {}

    def __enter__(self):
        for index, block in enumerate(self.blocks):
            self.handles.extend((
                block.register_forward_pre_hook(self._pre(index, "r")),
                block.self_attn.o_proj.register_forward_hook(self._out(index, "a")),
                block.post_attention_layernorm.register_forward_pre_hook(self._pre(index, "u")),
                block.mlp.down_proj.register_forward_hook(self._out(index, "m")),
                block.register_forward_hook(self._out(index, "h")),
            ))
        return self

    def __exit__(self, *unused):
        for handle in self.handles:
            handle.remove()

    def _pre(self, index, name):
        def hook(_module, inputs):
            self.values.setdefault(index, {})[name] = inputs[0].detach()
        return hook

    def _out(self, index, name):
        def hook(_module, _inputs, output):
            self.values.setdefault(index, {})[name] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook

    def take(self) -> dict[int, dict[str, torch.Tensor]]:
        result, self.values = self.values, {}
        required = {"r", "a", "u", "m", "h"}
        if len(result) != len(self.blocks) or any(set(row) != required for row in result.values()):
            raise ValueError("incomplete Qwen hook capture")
        return result


def _generated_rows(capture: dict[int, dict[str, torch.Tensor]], position: int) -> tuple[dict[int, dict[str, torch.Tensor]], torch.Tensor]:
    """Move only durable BF16 vectors plus FP32 geometry off GPU after each decode."""
    states = {layer: {name: _native_cpu(values[name][0, position])
                      for name in ("h", "a", "m")}
              for layer, values in capture.items()}
    geometry = torch.cat([geometry_channels(values["r"][0, position], values["a"][0, position],
                                            values["m"][0, position], values["u"][0, position],
                                            values["h"][0, position]).reshape(-1)
                          for _, values in sorted(capture.items())]).cpu()
    return states, geometry


def _prompt_geometry(capture: dict[int, dict[str, torch.Tensor]], position: int) -> torch.Tensor:
    return torch.cat([geometry_channels(values["r"][0, position], values["a"][0, position],
                                        values["m"][0, position], values["u"][0, position],
                                        values["h"][0, position]).reshape(-1)
                      for _, values in sorted(capture.items())]).cpu()


def _prompt_h(capture: dict[int, dict[str, torch.Tensor]], position: int) -> torch.Tensor:
    return torch.stack([_native_cpu(values["h"][0, position])
                        for _, values in sorted(capture.items())])


def _native_cpu(value: torch.Tensor) -> torch.Tensor:
    if value.dtype != torch.bfloat16:
        raise ValueError("collector requires native BF16 hook states; refusing a conversion")
    return value.detach().to("cpu")


def _prompt_raw(capture: dict[int, dict[str, torch.Tensor]], positions: list[int], name: str, hidden: int) -> torch.Tensor:
    if not positions:
        return torch.empty((0, len(capture), hidden), dtype=torch.uint16)
    return torch.stack([torch.stack([bf16_bits(_native_cpu(values[name][0, pos]))
                                     for _, values in sorted(capture.items())]) for pos in positions])


def _forward_geometry(capture: dict[int, dict[str, torch.Tensor]]) -> torch.Tensor:
    """All prompt rows are scalar-only, so persisting them is inexpensive."""
    return torch.stack([geometry_channels(values["r"], values["a"], values["m"], values["u"], values["h"])[0]
                        for _, values in sorted(capture.items())], dim=1).cpu()


def _moments(width: int) -> dict[str, Welford]:
    return {phase: Welford.empty(width) for phase in ("Q", "R", "A", "G_unsegmented")}


def _moments_tensor(moments: dict[str, Welford]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (torch.tensor([moments[p].n for p in moments], dtype=torch.int64),
            torch.stack([moments[p].mean for p in moments]), torch.stack([moments[p].m2 for p in moments]))


def _output_row(logits: torch.Tensor, chosen: int, greedy: bool, temperature: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = logits.detach().float()
    if not torch.isfinite(logits).all():
        raise ValueError("nonfinite output logits")
    logz = torch.logsumexp(logits, 0)
    logp = logits - logz
    entropy = -(logp.exp() * logp).sum()
    values, ids = torch.topk(logits, min(20, logits.numel()))
    ids = torch.nn.functional.pad(ids.to(torch.uint32), (0, 20 - len(ids)))
    values = torch.nn.functional.pad(values, (0, 20 - len(values)), value=float("-inf"))
    policy_logp = torch.zeros_like(logz) if greedy else torch.log_softmax(logits / temperature, 0)[chosen]
    metrics = torch.stack((logz, entropy, logp[chosen], policy_logp))
    return ids.cpu(), values.cpu(), metrics.cpu()


def render_prompt(tokenizer, question: str) -> tuple[torch.Tensor, list[int], str]:
    """Render once, then prove template IDs equal offset-tokenized rendered text."""
    messages = [{"role": "user", "content": question}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    template_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    if ids != template_ids:
        raise ValueError("chat template token IDs differ from rendered prompt tokenization")
    start = rendered.find(question)
    if start < 0 or rendered.find(question, start + 1) >= 0:
        raise ValueError("question is not a unique substring of rendered chat prompt")
    end = start + len(question)
    q_positions = [i for i, (left, right) in enumerate(encoded["offset_mapping"])
                   if left >= start and right <= end and right > left]
    if not q_positions:
        raise ValueError("no fully-contained question tokens after chat template")
    return torch.tensor(ids, dtype=torch.long), q_positions, rendered


def _input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def collect_case(model, tokenizer, case: dict, args, model_spec: dict, output: Path, *, is_raw: bool) -> Path:
    """Generate one answer and atomically archive the agreed bounded observations."""
    prompt, q_positions, rendered = render_prompt(tokenizer, case["text"])
    if prompt.numel() > args.max_prompt_tokens:
        raise ValueError(f"prompt has {prompt.numel()} tokens > --max-prompt-tokens")
    context = getattr(model.config, "max_position_embeddings", None)
    if context is not None and prompt.numel() + args.max_new_tokens > context:
        raise ValueError(f"prompt tokens + --max-new-tokens exceeds model context ({context})")
    device = _input_device(model)
    prompt = prompt.unsqueeze(0).to(device)
    mask = torch.ones_like(prompt)
    close_ids = tokenizer("</think>", add_special_tokens=False)["input_ids"]
    if not close_ids:
        raise ValueError("tokenizer cannot encode </think>")
    eos = tokenizer.eos_token_id
    generated_geometry: list[torch.Tensor] = []
    chosen_ids: list[int] = []
    top_ids: list[torch.Tensor] = []; top_logits: list[torch.Tensor] = []; output_metrics: list[torch.Tensor] = []
    progress_started = time.monotonic(); progress_last = progress_started
    blocks = len(getattr(model.model, "layers"))
    geometry_moments = _moments(blocks * 13)
    h_moments = _moments(blocks * model.config.hidden_size)
    g_geometry, g_h = Welford.empty(blocks * 13), Welford.empty(blocks * model.config.hidden_size)
    selector = StreamingSelector(seed=args.selection_seed, run_id=case["case_id"], raw=is_raw)
    pending = deque(); hit_count = 0; first_hit = None; last_content_h = None
    def commit(frame):
        nonlocal last_content_h
        if frame["service"] or not tokenizer.decode([frame["token"]], skip_special_tokens=False).strip():
            return
        g_geometry.update(frame["geometry"].reshape(1, -1))
        h = torch.stack([frame["states"][layer]["h"] for layer in range(blocks)]).reshape(1, -1)
        g_h.update(h)
        phase = "R" if first_hit is None or frame["position"] < first_hit else "A"
        geometry_moments[phase].update(frame["geometry"].reshape(1, -1)); h_moments[phase].update(h)
        selector.add(frame)
        last_content_h = h.reshape(blocks, model.config.hidden_size)
    with torch.inference_mode(), HookCapture(model) as hooks:
        response = model(input_ids=prompt, attention_mask=mask, use_cache=True, return_dict=True)
        prompt_capture = hooks.take()
        if len(prompt_capture) != blocks:
            raise ValueError("model block count changed during collection")
        prompt_geometry = _forward_geometry(prompt_capture)
        raw_q = raw_prompt_positions(q_positions) if is_raw else []
        raw_q_states = {name: _prompt_raw(prompt_capture, raw_q, name, model.config.hidden_size) for name in ("h", "a", "m")} if is_raw else {}
        for pos in q_positions:
            geometry_moments["Q"].update(prompt_geometry[pos].reshape(1, -1))
            h_moments["Q"].update(_prompt_h(prompt_capture, pos).reshape(1, -1))
        prompt_question_h = _prompt_h(prompt_capture, q_positions[-1])
        prompt_end_h = _prompt_h(prompt_capture, prompt.shape[1] - 1)
        del prompt_capture
        cache = response.past_key_values
        logits = response.logits[0, -1]
        seed = int.from_bytes(_hash("sample-v1", args.selection_seed, case["case_id"])[:8], "little")
        generator = torch.Generator(device=device).manual_seed(seed)
        for step in range(args.max_new_tokens):
            greedy = args.do_sample is False
            token = int(torch.argmax(logits).item()) if greedy else int(torch.multinomial(torch.softmax(logits.float() / args.temperature, -1), 1, generator=generator).item())
            row = _output_row(logits, token, greedy, args.temperature)
            top_ids.append(row[0]); top_logits.append(row[1]); output_metrics.append(row[2]); chosen_ids.append(token)
            token_input = torch.tensor([[token]], dtype=torch.long, device=device)
            response = model(input_ids=token_input, past_key_values=cache, use_cache=True, return_dict=True)
            states, geometry = _generated_rows(hooks.take(), 0)
            frame = {"position": len(chosen_ids) - 1, "token": token, "states": states, "geometry": geometry, "service": eos is not None and token == eos}
            pending.append(frame); generated_geometry.append(geometry)
            marker = list(close_ids)
            if len(chosen_ids) >= len(marker) and chosen_ids[-len(marker):] == marker:
                hit_count += 1; first_hit = len(chosen_ids) - len(marker) if first_hit is None else first_hit
                for old in list(pending)[-len(marker):]: old["service"] = True
            if len(pending) > len(marker) - 1:
                commit(pending.popleft())
            cache, logits = response.past_key_values, response.logits[0, -1]
            if eos is not None and token == eos:
                break
            now = time.monotonic()
            if now - progress_last >= 5:
                elapsed = max(now - progress_started, 1e-9)
                print(json.dumps({"current": case["case_id"], "generated_tokens": len(chosen_ids),
                                  "seconds": round(elapsed,1), "tokens_per_second": round(len(chosen_ids)/elapsed,3),
                                  **getattr(args, "_outer_progress", {})}, ensure_ascii=False), flush=True)
                progress_last = now
    while pending: commit(pending.popleft())
    roles, phase_status = segment_generation(chosen_ids, close_ids, eos_id=eos)
    if phase_status != "confirmed":
        geometry_moments["G_unsegmented"], h_moments["G_unsegmented"] = g_geometry, g_h
        geometry_moments["R"], geometry_moments["A"] = Welford.empty(blocks * 13), Welford.empty(blocks * 13)
        h_moments["R"], h_moments["A"] = Welford.empty(blocks * model.config.hidden_size), Welford.empty(blocks * model.config.hidden_size)
    sparse = selector.sparse()
    sparse_rows = torch.stack([torch.stack([bf16_bits(frame["states"][layer]["h"]) for layer in LAYERS]) for frame, _ in sparse]) if sparse else torch.empty((0, len(LAYERS), model.config.hidden_size), dtype=torch.uint16)
    anchor_h = last_content_h if last_content_h is not None else torch.zeros((blocks, model.config.hidden_size), dtype=torch.bfloat16)
    anchors = torch.stack([bf16_bits(row) for row in (prompt_question_h, prompt_end_h, anchor_h)])
    datasets = {
        "generated_ids": torch.tensor(chosen_ids, dtype=torch.int32),
        "generated_roles": torch.tensor([{"R": 1, "A": 2, "service": 3, "unknown": 4}[x] for x in roles], dtype=torch.uint8),
        "top_ids": torch.stack(top_ids) if top_ids else torch.empty((0, 20), dtype=torch.uint32),
        "top_logits": torch.stack(top_logits) if top_logits else torch.empty((0, 20), dtype=torch.float32),
        "output_metrics": torch.stack(output_metrics) if output_metrics else torch.empty((0, 4), dtype=torch.float32),
        "prompt_ids": prompt[0].cpu().to(torch.int32),
        "q_mask": torch.tensor([i in set(q_positions) for i in range(prompt.shape[1])], dtype=torch.uint8),
        "service_mask": torch.tensor([role == "service" for role in roles], dtype=torch.uint8),
        "prompt_geometry": prompt_geometry,
        "generated_geometry": torch.stack(generated_geometry).reshape(len(generated_geometry), blocks, 13) if generated_geometry else torch.empty((0, blocks, 13), dtype=torch.float32),
        "phase_geometry_n": _moments_tensor(geometry_moments)[0], "phase_geometry_mean": _moments_tensor(geometry_moments)[1], "phase_geometry_m2": _moments_tensor(geometry_moments)[2],
        "phase_h_n": _moments_tensor(h_moments)[0], "phase_h_mean": _moments_tensor(h_moments)[1], "phase_h_m2": _moments_tensor(h_moments)[2],
        "sparse_positions": torch.tensor([frame["position"] for frame, _ in sparse], dtype=torch.int32),
        "sparse_selection_roles": torch.tensor([role for _, role in sparse], dtype=torch.uint8),
        "sparse_h_bf16": sparse_rows,
        "anchors_h_bf16": anchors,
        "anchor_positions": torch.tensor([q_positions[-1], prompt.shape[1] - 1,
                                           selector.sparse()[-1][0]["position"] if last_content_h is not None else -1], dtype=torch.int32),
        "anchor_present": torch.tensor([1, 1, int(last_content_h is not None)], dtype=torch.uint8),
    }
    if is_raw:
        raw_g_frames = selector.raw_frames()
        datasets["raw_q_positions"] = torch.tensor(raw_q, dtype=torch.int32)
        datasets["raw_g_positions"] = torch.tensor([frame["position"] for frame in raw_g_frames], dtype=torch.int32)
        for name in ("h", "a", "m"):
            datasets[f"raw_q_{name}_bf16"] = raw_q_states[name]
            datasets[f"raw_g_{name}_bf16"] = torch.stack([torch.stack([bf16_bits(frame["states"][layer][name]) for layer in range(blocks)]) for frame in raw_g_frames]) if raw_g_frames else torch.empty((0, blocks, model.config.hidden_size), dtype=torch.uint16)
    manifest = {
        "format": FORMAT_VERSION, "case_id": case["case_id"], "family_id": case["family_id"], "variant_id": case["variant_id"],
        "input_sha256": hashlib.sha256(case["text"].encode()).hexdigest(), "model": model_spec,
        "max_new_tokens": args.max_new_tokens, "max_prompt_tokens": args.max_prompt_tokens,
        "do_sample": args.do_sample, "temperature": args.temperature,
        "layers": list(LAYERS), "geometry_channels": 13, "phase_status": phase_status,
        "selection_seed": args.selection_seed, "selection_algorithm": "first8-hash16-last8-v1",
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "store_sha256": hashlib.sha256(Path(__file__).with_name("_activation_store.py").read_bytes()).hexdigest(),
        "raw_subset": is_raw, "bf16_encoding": "little-endian uint16 native bfloat16 bits",
        "input_text": case["text"], "rendered_prompt": rendered, "decoded_output": tokenizer.decode(chosen_ids, skip_special_tokens=False),
        "prompt_tokens": int(prompt.shape[1]), "generated_tokens": len(chosen_ids),
        "stop_reason": "eos" if chosen_ids and eos is not None and chosen_ids[-1] == eos else "max_new_tokens",
        "sample_seed": seed,
        "peak_vector_frames": selector.peak_vector_frames,
    }
    return write_completed(output, manifest, datasets)


def collect_all(model, tokenizer, cases: list[dict], args, model_spec: dict) -> dict:
    """Resumable ordered loop.  A completed case is immutable; failures are logged."""
    root = args.out / "cases"; root.mkdir(parents=True, exist_ok=True)
    raw = raw_families(cases)
    completed = 0; skipped = 0; errors = 0; started = time.monotonic()
    error_path = args.out / "errors.jsonl"
    for index, case in enumerate(cases, 1):
        args._outer_progress = {"completed": completed, "skipped": skipped, "errors": errors, "total": len(cases)}
        path = root / f"{case['case_id'].replace(':', '__')}.h5"
        if path.exists():
            # Validate before trusting a resume.  Completed manifest parsing is intentionally strict.
            from research._activation_store import completed_manifest
            found = completed_manifest(path)
            expected = {"format": FORMAT_VERSION, "case_id": case["case_id"],
                        "input_sha256": hashlib.sha256(case["text"].encode()).hexdigest(), "model": model_spec,
                        "max_new_tokens": args.max_new_tokens, "max_prompt_tokens": args.max_prompt_tokens,
                        "selection_seed": args.selection_seed, "do_sample": args.do_sample, "temperature": args.temperature,
                        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        "store_sha256": hashlib.sha256(Path(__file__).with_name("_activation_store.py").read_bytes()).hexdigest(),
                        "raw_subset": case["family_id"] in raw and case["variant_id"] in {"original", "v01", "v02"}}
            if any(found.get(key) != value for key, value in expected.items()):
                raise ValueError(f"completed file does not validate: {path}")
            skipped += 1
        else:
            partial = path.with_suffix(".partial.h5")
            if partial.exists():
                # Preserve forensic evidence while allowing the whole input to restart.
                partial.replace(partial.with_name(partial.name + f".interrupted-{int(time.time())}"))
            try:
                collect_case(model, tokenizer, case, args, model_spec, path, is_raw=case["family_id"] in raw and case["variant_id"] in {"original", "v01", "v02"})
                completed += 1
            except KeyboardInterrupt:
                (args.out / "collection-status.json").write_text(json.dumps({"status":"interrupted", "current":case["case_id"]}), encoding="utf-8")
                raise
            except Exception as exc:
                if "nonfinite" in str(exc).lower():
                    (args.out / "collection-status.json").write_text(json.dumps({"status":"failed", "current":case["case_id"], "error":str(exc)}), encoding="utf-8")
                    raise
                errors += 1
                with error_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"case_id": case["case_id"], "type": type(exc).__name__, "message": str(exc)}, ensure_ascii=False) + "\n")
        elapsed = max(time.monotonic() - started, 1e-9)
        done = completed + skipped + errors
        print(json.dumps({"done": done, "total": len(cases), "completed": completed, "skipped": skipped, "errors": errors,
                          "seconds": round(elapsed, 1), "cases_per_second": round(done / elapsed, 4), "current": case["case_id"]}, ensure_ascii=False), flush=True)
        (args.out / "collection-status.json").write_text(json.dumps({"status":"running", "completed":completed, "skipped":skipped, "errors":errors, "total":len(cases), "current":case["case_id"]}), encoding="utf-8")
    result = {"completed": completed, "skipped": skipped, "errors": errors, "total": len(cases)}
    (args.out / "collection-status.json").write_text(json.dumps({"status":"completed" if not errors else "completed_with_errors", **result}), encoding="utf-8")
    return result
