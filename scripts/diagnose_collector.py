"""Диагностика AIMO: uv run python scripts/diagnose_collector.py --out D:/aimo-diagnostics/run1

Один файл для существующего checkout collector. Default: 5 часов, кешированные
веса, BF16, GPU 14/13 GiB, CPU offload 28 GiB. Запускать отдельно от collection.
--self-test: случайная крошечная Qwen на CPU, без сети и model_context.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
REVISION = "6e8885a6ff5c1dc5201574c8fd700323f23c25fa"
TARGET_INPUTS, TARGET_G, TARGET_SECONDS = 1507, 2048, 5 * 86400
SCHEMA = 3
PREFLIGHT_SCHEMA = 1
PREFLIGHT_MARKER = "diagnostic-preflight.json"
MODES = {
    "A": "forward + KV cache",
    "B": "A + production HookCapture/take",
    "C": "B + 13 geometry channels, original finite checks, geometry D2H",
    "D": "C + production _output_row",
    "E": "D + native BF16 h/a/m D2H (108 copies for 36 layers)",
    "F": "exact collect_case replay: phases/moments/selector/packing, no HDF5",
    "G": "F + production raw selector and h/a/m packing",
    "FULL": "unmodified production collect_case, natural EOS, HDF5/checksum",
    "AUX": "paired CPU Welford and actual packed HDF5 write/readback",
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(canonical(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def sealed(path, payload):
    atomic(path, {"payload": payload, "sha256": digest(payload)})


def unseal(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if digest(value["payload"]) != value["sha256"]:
        raise ValueError(f"checksum mismatch: {path}")
    return value["payload"]


def runtime_identity():
    versions = {}
    for package in ("torch", "transformers", "accelerate", "h5py", "tokenizers", "huggingface-hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": sys.version, "platform": sys.platform, "versions": versions,
            "thread_environment": {k: os.getenv(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}}


def source_identity():
    paths = [Path(__file__), *(ROOT / "research" / name for name in
                              ("_collector.py", "_activation_store.py", "_support.py", "features.py"))]
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths}


def study_identity(args):
    # Volatile paths and deadlines are deliberately absent.
    return {"schema": SCHEMA, "sources": source_identity(), "runtime": runtime_identity(),
            "inputs": {"originals": file_hash(args.originals), "variations": file_hash(args.variations)},
            "method": {"model": MODEL, "revision": REVISION, "dtype": "bfloat16", "template": "checkpoint chat",
                       "placements_gpu_gib": [14, 13], "cpu_gib": 28, "contexts": [0, 1024, 1984],
                       "decode_tokens": args.decode_tokens, "natural_tape_tokens": 16,
                       "synthetic_repeat": args.synthetic_repeat, "selection_seed": 7,
                       "warmups": 1, "repeats": 2, "full_max_prompt": 8192, "full_max_new": 2048}}


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, help="отдельный новый каталог; тот же каталог для resume")
    p.add_argument("--hours", type=float, default=5, help="общий бюджет с загрузками, default 5")
    p.add_argument("--decode-tokens", type=int, default=32)
    p.add_argument("--allow-download", action="store_true", help="разрешить скачать отсутствующие pinned model files; preflight скачивает только config/tokenizer")
    p.add_argument("--synthetic-repeat", action=argparse.BooleanOptionalAction, default=True,
                   help="повторять короткую natural tape до G1984+window; default enabled")
    p.add_argument("--originals", type=Path, default=ROOT / "data/collection_original_tasks.json")
    p.add_argument("--variations", type=Path, default=ROOT / "variations_tasks.json")
    p.add_argument("--self-test", action="store_true", help="CPU tests без model_context, сети и весов")
    p.add_argument("--preflight-only", action="store_true", help="проверить pinned config/tokenizer и все 1507 prompts без CUDA и весов")
    p.add_argument("--child", type=Path, help=argparse.SUPPRESS)
    return p


def hardware(out, gpu=False):
    result = {"disk_free_bytes": shutil.disk_usage(out).free, "runtime": runtime_identity()}
    try:
        import ctypes
        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in ("total", "available", "page_total", "page_available",
                                                       "virtual_total", "virtual_available", "extended")]
        m = Memory()
        m.length = ctypes.sizeof(m)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        result.update(ram_total_bytes=m.total, ram_available_bytes=m.available)
    except (AttributeError, OSError):
        pass
    if gpu:
        import torch
        result.update(torch_cuda=torch.version.cuda, threads=torch.get_num_threads())
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            result.update(gpu_name=torch.cuda.get_device_name(), vram_free_bytes=free, vram_total_bytes=total,
                          vram_allocated_bytes=torch.cuda.memory_allocated(), vram_reserved_bytes=torch.cuda.memory_reserved(),
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return result


def cell(name, mode, percentile="P50", offset=1024, placement=14, stage="matrix", threads=None):
    return dict(name=name, mode=mode, percentile=percentile, offset=offset, placement=placement,
                stage=stage, threads=threads)


def make_cells():
    cells = [cell("matrix_" + mode, mode) for mode in "ABCDEFG"]
    cells += [cell("candidate_F_threads1", "F", threads=1)]
    cells += [cell("full_P50_nonraw", "FULL", offset=0, stage="full"),
              cell("full_P90_raw", "FULL", "P90", offset=0, stage="full")]
    cells += [cell(f"window_{p}_G{g}_{mode}", mode, p, g)
              for p in ("P10", "P90") for g in (0, 1984) for mode in ("A", "F")]
    cells += [cell("max_prompt_stress", "G", "max", 0),
              cell("cpu_hdf_aux", "AUX", stage="aux"),
              cell("placement13_A", "A", placement=13, stage="placement")]
    return cells


def compatible_full_eta(rows):
    # Never average ablations; EOS-short output cannot prove all-2048 throughput.
    result = []
    for row in rows:
        actual = row.get("actual", {})
        if row.get("mode") != "FULL" or row.get("status") != "success":
            continue
        n, seconds = actual.get("generated_tokens", 0), actual.get("end_to_end_seconds", 0)
        item = {"name": row["name"], "placement": row["placement"], "generated_tokens": n,
                "stop_reason": actual.get("stop_reason"), "seconds_per_input": seconds,
                "observed_tokens_per_second_including_prefill_write": n / seconds if seconds else None,
                "conditional_days_all2048": None}
        if n == 2048 and actual.get("stop_reason") == "max_new_tokens" and seconds > 0:
            item["conditional_days_all2048"] = TARGET_INPUTS * seconds / 86400
            item["caveat"] = "conditional on sampled prompt/raw configuration representing all inputs"
        else:
            item["caveat"] = "EOS-short/incomplete: insufficient evidence for 2048-token input cost"
        result.append(item)
    return result


def report(out, state):
    rows = state["cells"]
    forecasts = compatible_full_eta(rows)
    lines = ["# Диагностика сборщика AIMO", "",
             f"Цель: 1507 × 2048 за 5 суток; минимум {TARGET_INPUTS * TARGET_G / TARGET_SECONDS:.3f} токена/с",
             f"до prefill/записи, или {TARGET_SECONDS / TARGET_INPUTS:.2f} с на весь input.",
             f"Израсходовано {state.get('spent_seconds', 0) / 3600:.2f} ч из {state['budget_seconds'] / 3600:.2f} ч.", "",
             "| Ячейка | GPU GiB | Статус | Decode с/токен | Prefill с | FULL с |",
             "|---|---:|---|---:|---:|---:|"]
    for row in rows:
        actual = row.get("actual", {})
        ms = actual.get("measurements", [])
        rate = statistics.median([m["decode_seconds"] / m["tokens"] for m in ms]) if ms else None
        prefill = statistics.median([m["prefill_seconds"] for m in ms]) if ms else None
        fmt = lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else "—"
        lines.append(f"| {row['name']} | {row['placement']} | {row['status']} | {fmt(rate)} | {fmt(prefill)} | {fmt(actual.get('end_to_end_seconds'))} |")
    lines += ["", "A–G используют одни token IDs. G0/1024/1984 — продолжение после полного prompt.",
              "KV восстанавливается одним prefill: его округления могут отличаться от последовательного decode.",
              "F/G запускают настоящие операции production, но phase/selector history начинается в окне: короткий replay",
              "не проверяет заполненные raw-буферы и накопленную историю G1024/1984. Это проверяет только FULL.",
              "Повторение короткой natural tape — явно синтетическая нагрузка.", ""]
    lines += [f"- {mode}: {description}." for mode, description in MODES.items()]
    lines += ["", "## Измеренная стоимость", ""]
    baseline = {r["mode"]: r for r in rows if r["name"].startswith("matrix_") and r["status"] == "success"}
    differences = []
    for left, right in zip("ABCDEF", "BCDEFG"):
        if left in baseline and right in baseline:
            rates = [statistics.median(m["decode_seconds"] / m["tokens"] for m in baseline[mode]["actual"]["measurements"])
                     for mode in (left, right)]
            differences.append({"increment": f"{left}→{right}", "seconds_per_token": rates[1] - rates[0]})
    for d in differences:
        lines.append(f"- {d['increment']}: {d['seconds_per_token']:+.4f} с/токен.")
    lines.append("Разности включают шум порядка запусков; F/G включают упаковку окна. Отрицательная разность не доказывает ускорение."
                 if differences else "Недостаточно сопоставимых завершённых ячеек для вывода об узком месте.")
    candidate = next((r for r in rows if r["name"] == "candidate_F_threads1"), {})
    eq = candidate.get("actual", {}).get("equivalence")
    lines += ["", "Кандидат: CPU threads=1 на той же F-нагрузке. " +
              (f"Эквивалентность данных: {eq['passed']}. Подробности по каждому dataset в JSON." if eq else "Эквивалентность пока не установлена."),
              "Packed D2H и пакетные finite checks остаются гипотезами, их ускорение здесь не утверждается.",
              "", "## Прогноз пяти дней", ""]
    for forecast in forecasts:
        days = forecast["conditional_days_all2048"]
        lines.append(f"- {forecast['name']}: {forecast['generated_tokens']} токенов, {forecast['seconds_per_input']:.1f} с; " +
                     (f"условно {days:.2f} суток на 1507 inputs с таким же prompt/raw режимом." if days else
                      "EOS/недостаточно данных: стоимость всех 2048 токенов неизвестна."))
    if not forecasts:
        lines.append("Нет завершённого FULL: достижимость 5 суток неизвестна.")
    lines += ["", "FULL включает prompt processing, decode, packing, HDF5 и checksum; загрузка указана отдельно.",
              "Среднее A–G никогда не используется для ETA. P50/P90 не дают репрезентативной оценки всех inputs.",
              "HDF5 timing включает файловый кеш: это не физическая пропускная способность диска.",
              "Внутренние wall-times inclusive, неаддитивны, без дополнительных CUDA synchronize.",
              "Пропуски, ошибки и число повторов записаны в JSON. Прерванные FULL не являются результатом."]
    summary = out / "summary.md"
    temp = summary.with_suffix(".md.tmp")
    temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temp.replace(summary)
    atomic(out / "report.json", {"schema": SCHEMA, "state": state, "mode_definitions": MODES,
                                "conditional_full_forecasts": forecasts, "measured_mode_differences": differences})


def emit(event, **fields):
    print(canonical({"diagnostic_event": event, **fields}), flush=True)


def sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def patched(obj, **replacements):
    originals = {name: getattr(obj, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(obj, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(obj, name, value)


def tensor_hash(tensor):
    import torch
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.uint16)
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def compare_datasets(left, right):
    import torch
    if set(left) != set(right):
        raise ValueError("candidate dataset keys differ")
    details = {}
    for key in left:
        a, b = left[key], right[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"candidate schema differs: {key}")
        exact = torch.equal(a, b)
        close = bool(torch.allclose(a, b, atol=2e-5, rtol=2e-5)) if a.is_floating_point() else exact
        delta = float((a.float() - b.float()).abs().max()) if a.numel() and a.is_floating_point() else 0.0
        # Equal padded -inf top-logit values have zero difference, not NaN.
        if exact:
            delta = 0.0
        details[key] = {"exact": exact, "within_tolerance": close, "max_abs": delta}
        if not close:
            raise ValueError(f"candidate numerical mismatch: {key}, max_abs={delta}")
    return {"passed": True, "atol": 2e-5, "rtol": 2e-5, "datasets": details,
            "coverage": "all 13 geometry channels; h BF16 bits; phase h/geometry n,mean,M2; positions; top logits/IDs/output metrics",
            "unaffected": "a/m copying algorithm unchanged; full logit endpoints and argmax compared separately"}


def make_prefix(prompt, tape, offset, count, context_limit):
    import torch
    if len(tape) < offset + count:
        raise OverflowError("natural tape too short and synthetic repeat disabled")
    prefix = torch.cat((prompt, torch.tensor(tape[:offset], dtype=torch.long)))
    if prefix.numel() + count > context_limit:
        raise OverflowError(f"prompt {prompt.numel()} + G{offset} + window {count} exceeds context {context_limit}")
    return prefix, tape[offset:offset + count]


def staged_window(model, prompt, tokens, mode):
    """A..E: C has no Welford; E does not recompute geometry."""
    import torch
    from research import _collector as c
    rank = "ABCDE".index(mode)
    device = c._input_device(model)
    counters = {"forward": 0, "take": 0, "geometry": 0, "output_row": 0, "native_copies": 0, "welford": 0}
    endpoint_digest, argmax = hashlib.sha256(), []
    with torch.inference_mode(), (c.HookCapture(model) if rank else nullcontext(None)) as hooks:
        sync()
        start = time.perf_counter()
        ids = prompt.unsqueeze(0).to(device)
        response = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True, return_dict=True)
        if hooks:
            hooks.take()
        cache, logits = response.past_key_values, response.logits[0, -1]
        sync()
        prefill = time.perf_counter() - start
        endpoint_digest.update(tensor_hash(logits).encode())
        argmax.append(int(logits.argmax()))
        sync()
        start = time.perf_counter()
        for token in tokens:
            if rank >= 3:
                c._output_row(logits, token, True, 1.0)
                counters["output_row"] += 1
            token_input = torch.tensor([[token]], device=device)
            response = model(input_ids=token_input, past_key_values=cache, use_cache=True, return_dict=True)
            counters["forward"] += 1
            if hooks:
                capture = hooks.take()
                counters["take"] += 1
                if rank >= 2:
                    geometry = c._prompt_geometry(capture, 0)
                    counters["geometry"] += len(capture)
                if rank >= 4:
                    states = {layer: {name: c._native_cpu(values[name][0, 0]) for name in ("h", "a", "m")}
                              for layer, values in capture.items()}
                    counters["native_copies"] += len(capture) * 3
                del capture
            cache, logits = response.past_key_values, response.logits[0, -1]
        sync()
        decode = time.perf_counter() - start
        endpoint_digest.update(tensor_hash(logits).encode())
        argmax.append(int(logits.argmax()))
    return {"tokens": len(tokens), "prefill_seconds": prefill, "decode_seconds": decode,
            "logit_endpoints_sha256": endpoint_digest.hexdigest(), "argmax_endpoints": argmax, "counters": counters}


def replay_window(model, tokenizer, case, prompt, question_positions, rendered, tokens, raw, model_spec):
    """Exact production pipeline, forced tape, in-memory writer; no copied F/G logic.

    Only collection's torch.argmax is intercepted, never global torch. Output rows
    see original logits. Synthetic tape excludes EOS. Pending/marker/whitespace,
    phase moments, sparse/raw selection and final packing are all production code.
    """
    import torch
    from research import _collector as c
    base_torch, old_row = c.torch, c._output_row
    position, first_output = 0, None
    captured, last_response = {}, []
    endpoint_digest, argmax = hashlib.sha256(), []
    class ForcedTorch:
        def __getattr__(self, name):
            return getattr(base_torch, name)
        def argmax(self, logits):
            nonlocal position
            token = tokens[position]
            position += 1
            return base_torch.tensor(token)
    class TapeTokenizer:
        eos_token_id = None
        def __getattr__(self, name):
            return getattr(tokenizer, name)
        def __call__(self, *args, **kwargs):
            return tokenizer(*args, **kwargs)
    class ModelProxy:
        def __getattr__(self, name):
            return getattr(model, name)
        def __call__(self, *args, **kwargs):
            response = model(*args, **kwargs)
            last_response[:] = [response]
            return response
    def row(logits, chosen, greedy, temperature):
        nonlocal first_output
        if first_output is None:
            sync()
            captured["prefill_seconds"] = time.perf_counter() - started
            endpoint_digest.update(tensor_hash(logits).encode())
            argmax.append(int(logits.argmax()))
            sync()
            first_output = time.perf_counter()
        return old_row(logits, chosen, greedy, temperature)
    def writer(path, manifest, datasets):
        sync()
        captured.update(decode_seconds=time.perf_counter() - first_output, datasets=datasets, manifest=manifest)
        return path
    args = SimpleNamespace(max_prompt_tokens=max(2048, int(prompt.numel())), max_new_tokens=len(tokens),
                           do_sample=False, temperature=1.0, selection_seed=7)
    with patched(c, torch=ForcedTorch(), render_prompt=lambda *_: (prompt, question_positions, rendered),
                 _output_row=row, write_completed=writer):
        sync()
        started = time.perf_counter()
        c.collect_case(ModelProxy(), TapeTokenizer(), case, args, model_spec, Path("in-memory-only.h5"), is_raw=raw)
    endpoint_digest.update(tensor_hash(last_response[0].logits[0, -1]).encode())
    argmax.append(int(last_response[0].logits[0, -1].argmax()))
    captured.update(tokens=len(tokens), logit_endpoints_sha256=endpoint_digest.hexdigest(), argmax_endpoints=argmax,
                    exact_production_replay=True, raw=raw)
    return captured


def full_run(model, tokenizer, case, model_spec, artifact, raw):
    from research import _collector as c
    from research._activation_store import completed_manifest
    durations, originals = {}, {}
    for name in ("_generated_rows", "_forward_geometry", "_prompt_h", "_output_row", "write_completed"):
        old = getattr(c, name)
        originals[name] = old
        def wrapper(*args, _old=old, _name=name, **kwargs):
            started = time.perf_counter()
            try:
                return _old(*args, **kwargs)
            finally:
                durations[_name] = durations.get(_name, 0) + time.perf_counter() - started
        setattr(c, name, wrapper)
    args = SimpleNamespace(max_prompt_tokens=8192, max_new_tokens=2048, do_sample=False, temperature=1.0, selection_seed=7)
    try:
        sync()
        started = time.perf_counter()
        output = c.collect_case(model, tokenizer, case, args, model_spec, artifact, is_raw=raw)
        sync()
        elapsed = time.perf_counter() - started
    finally:
        for name, old in originals.items():
            setattr(c, name, old)
    started = time.perf_counter()
    manifest = completed_manifest(output)
    verify_seconds = time.perf_counter() - started
    return {"end_to_end_seconds": elapsed, "generated_tokens": manifest["generated_tokens"],
            "prompt_tokens": manifest["prompt_tokens"], "stop_reason": manifest["stop_reason"],
            "phase_status": manifest["phase_status"], "raw": raw,
            "artifact_name": output.name, "artifact_sha256": file_hash(output),
            "extra_validation_seconds": verify_seconds, "inclusive_wall_stage_seconds_nonadditive": durations,
            "timing_scope": "synchronized entire collect_case incl write_completed checksum/readback; excludes load"}


def paired_welford(h_rows):
    import torch
    from research._activation_store import Welford
    default, baseline, records = torch.get_num_threads(), None, []
    try:
        for threads in dict.fromkeys((default, 1, 4)):
            torch.set_num_threads(threads)
            timings = []
            for repeat in range(3):
                w = Welford.empty(h_rows.shape[1])
                started = time.perf_counter()
                for row in h_rows:
                    w.update(row.reshape(1, -1))
                elapsed = time.perf_counter() - started
                if repeat:
                    timings.append(elapsed)
            if baseline is None:
                baseline = w
            fp64 = h_rows.double()
            reference_mean = fp64.mean(0)
            reference_m2 = ((fp64 - reference_mean) ** 2).sum(0)
            equal = w.n == baseline.n and torch.equal(w.mean, baseline.mean) and torch.equal(w.m2, baseline.m2)
            fp64_ok = torch.allclose(w.mean.double(), reference_mean, atol=2e-5, rtol=2e-5) and torch.allclose(w.m2.double(), reference_m2, atol=2e-4, rtol=2e-4)
            if not equal or not fp64_ok:
                raise ValueError("CPU Welford numerical equivalence failed")
            records.append({"threads": threads, "count": w.n, "seconds": timings, "paired_exact": equal,
                            "fp64_reference_passed": bool(fp64_ok), "mean_sha256": tensor_hash(w.mean), "m2_sha256": tensor_hash(w.m2),
                            "mean_fp64_max_abs": float((w.mean.double() - reference_mean).abs().max()),
                            "m2_fp64_max_abs": float((w.m2.double() - reference_m2).abs().max())})
    finally:
        torch.set_num_threads(default)
    return records


def auxiliary(datasets, artifact):
    import torch
    from research._activation_store import write_completed, completed_manifest
    # All blocks (36 on the target), unlike sparse_h's 12-layer width. Repeat
    # three actual captured anchors to measure production-width CPU updates.
    anchors = datasets["anchors_h_bf16"].contiguous().view(torch.bfloat16)
    h_rows = anchors.reshape(anchors.shape[0], -1).repeat(16, 1)
    records = paired_welford(h_rows)
    started = time.perf_counter()
    packed = {key: value.contiguous() for key, value in datasets.items()}
    packing_seconds = time.perf_counter() - started
    manifest = {"diagnostic": True, "source": "exact production F replay packed datasets"}
    started = time.perf_counter()
    write_completed(artifact, manifest, packed)
    write_seconds = time.perf_counter() - started
    started = time.perf_counter()
    if completed_manifest(artifact) != manifest:
        raise ValueError("HDF5 readback differs")
    return {"welford": records, "captured_shape": list(h_rows.shape),
            "h_source": "three real all-layer h anchors replayed 16 times; synthetic CPU workload, original BF16 values",
            "contiguous_packing_seconds": packing_seconds, "write_checksum_readback_seconds": write_seconds,
            "additional_validated_readback_seconds": time.perf_counter() - started, "artifact_name": artifact.name,
            "artifact_sha256": file_hash(artifact), "file_bytes": artifact.stat().st_size,
            "scope": "application HDF5/checksum path, buffered filesystem, not physical disk bandwidth"}


def prepare_inputs(spec, tokenizer, model_spec):
    from research._activation_store import plan_cases
    from research._collector import render_prompt
    cases = plan_cases(json.loads(Path(spec["originals"]).read_text(encoding="utf-8")),
                       json.loads(Path(spec["variations"]).read_text(encoding="utf-8")))
    if len(cases) != TARGET_INPUTS:
        raise ValueError(f"expected {TARGET_INPUTS} label-free inputs, got {len(cases)}")
    rows, by_id = [], {case["case_id"]: case for case in cases}
    for case in cases:
        prompt, _, _ = render_prompt(tokenizer, case["text"])
        rows.append({"case_id": case["case_id"], "tokens": int(prompt.numel())})
    rows.sort(key=lambda row: (row["tokens"], row["case_id"]))
    selected = {key: rows[round(q * (len(rows) - 1))] for key, q in {"P10": .1, "P50": .5, "P90": .9, "max": 1}.items()}
    identity = {"model_spec": model_spec, "template_sha256": digest(tokenizer.chat_template),
                "token_inventory": rows, "selection": selected}
    path = Path(spec["out"]) / "input-selection.json"
    if path.exists():
        if unseal(path) != identity:
            raise ValueError("actual tokenizer/template/selection fingerprint changed; use a new --out")
    else:
        sealed(path, identity)
    return {key: by_id[row["case_id"]] for key, row in selected.items()}, identity


def preflight_inputs(args, out):
    """Verify the exact tokenizer and all inputs before allocating either GPU placement."""
    from research import _support as io
    context = SimpleNamespace(model=MODEL, revision=REVISION, dtype="bfloat16", device="cuda", layers=None,
                              max_length=None, template="chat", allow_download=args.allow_download)
    tokenizer, model_spec = io.model_context(context)
    _, setup = prepare_inputs({"originals": str(args.originals.resolve()), "variations": str(args.variations.resolve()),
                               "out": str(out)}, tokenizer, model_spec)
    return model_spec, setup


def write_preflight_failure(out, exc):
    out.mkdir(parents=True, exist_ok=True)
    message = f"{type(exc).__name__}: {exc}"
    (out / "summary.md").write_text("# Диагностика сборщика AIMO\n\n## Preflight не пройден\n\n"
                                    f"{message}\n\nCUDA, веса и обе GPU-разметки не запускались. "
                                    "Исправьте tokenizer/input contract и повторите с новым или очищенным diagnostic --out.\n",
                                    encoding="utf-8")
    print(f"Preflight failed before GPU/model loading: {message}", file=sys.stderr, flush=True)


def verify_preflight_marker(out, identity, study_id):
    marker = out / PREFLIGHT_MARKER
    if not marker.exists():
        return False
    payload = unseal(marker)
    selection = out / "input-selection.json"
    if (payload.get("schema") != PREFLIGHT_SCHEMA or payload.get("study_id") != study_id
            or payload.get("identity") != identity or not selection.exists()
            or payload.get("input_selection_sha256") != file_hash(selection)):
        raise ValueError("preflight marker/input selection differs; use a new --out")
    return True


def write_preflight_marker(out, identity, study_id):
    selection = out / "input-selection.json"
    if not selection.exists():
        raise ValueError("preflight did not produce input selection")
    sealed(out / PREFLIGHT_MARKER, {"schema": PREFLIGHT_SCHEMA, "study_id": study_id,
                                    "identity": identity, "input_selection_sha256": file_hash(selection)})


def natural_tape(model, tokenizer, case, needed, synthetic):
    import torch
    from research._collector import render_prompt, _input_device
    prompt, _, _ = render_prompt(tokenizer, case["text"])
    device, tokens, stop = _input_device(model), [], "short_baseline_limit"
    sync()
    started = time.perf_counter()
    with torch.inference_mode():
        ids = prompt.unsqueeze(0).to(device)
        response = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True, return_dict=True)
        for _ in range(16):
            logits = response.logits[0, -1]
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite natural tape logits")
            token = int(logits.argmax())
            if token == tokenizer.eos_token_id:
                stop = "eos"
                break
            tokens.append(token)
            response = model(input_ids=torch.tensor([[token]], device=device), past_key_values=response.past_key_values,
                             use_cache=True, return_dict=True)
    sync()
    elapsed = time.perf_counter() - started
    natural = list(tokens)
    fallback = False
    if not tokens and synthetic:
        tokens = [x for x in tokenizer(" 1", add_special_tokens=False)["input_ids"] if x != tokenizer.eos_token_id]
        fallback = True
    if not tokens:
        raise OverflowError("empty natural continuation; no usable load tape")
    if synthetic and len(tokens) < needed:
        tokens = (tokens * ((needed + len(tokens) - 1) // len(tokens)))[:needed]
    return {"case_id": case["case_id"], "natural_ids": natural, "ids": tokens, "natural_stop": stop,
            "synthetic_load": len(tokens) > len(natural) or fallback, "empty_natural_fallback": fallback,
            "construction": "repeat short greedy baseline token IDs; never repeat/truncate the question",
            "natural_generation_seconds": elapsed, "eos_in_tape": tokenizer.eos_token_id in tokens}


def get_tape(spec, model, tokenizer, case, setup_id):
    path = Path(spec["out"]) / "tapes" / (digest(case["case_id"])[:16] + ".json")
    if path.exists():
        tape = unseal(path)
        if tape["study_id"] != spec["study_id"] or tape["setup_id"] != setup_id:
            raise ValueError("tape fingerprint differs")
    else:
        tape = natural_tape(model, tokenizer, case, 1984 + spec["decode_tokens"], spec["synthetic_repeat"])
        tape.update(study_id=spec["study_id"], setup_id=setup_id)
        sealed(path, tape)
    return tape, path


def cell_payload(spec, definition, setup_id, actual=None, status="success", reason=None, artifacts=None):
    signature = {"study_id": spec["study_id"], "cell": definition, "setup_id": setup_id}
    payload = {**definition, "study_id": spec["study_id"], "fingerprint": digest(signature),
               "signature": signature, "status": status, "artifacts": artifacts or {}}
    if actual is not None:
        payload["actual"] = actual
    if reason:
        payload["reason"] = reason
    return payload


def publish_cell(spec, payload):
    path = Path(spec["attempt_dir"]) / (payload["name"] + ".json")
    sealed(path, payload)
    emit("cell_result", relative_result=path.relative_to(Path(spec["out"])).as_posix())


def stage_remaining(previous, following, started, consumed, limits, at):
    """Returning to matrix after FULL must retain the earlier matrix expenditure."""
    if previous is not None:
        consumed[previous] = consumed.get(previous, 0) + max(0, at - started)
    return max(0, limits.get(following, 0) - consumed.get(following, 0))


def child_group(spec):
    import torch
    from research import _support as io, _collector as c
    from research._activation_store import require_cuda
    require_cuda()  # Before any model/tokenizer access.
    out = Path(spec["out"])
    group_deadline = time.monotonic() + spec["group_seconds"]
    emit("stage", stage="setup", seconds=min(1200, spec["group_seconds"]))
    args = SimpleNamespace(model=MODEL, revision=REVISION, dtype="bfloat16", device="cuda", layers=None,
                           max_length=None, template="chat", allow_download=spec["allow_download"])
    started = time.perf_counter()
    tokenizer, model_spec = io.model_context(args)
    picks, setup = prepare_inputs(spec, tokenizer, model_spec)
    setup_id = digest(setup)
    model = io.load_backbone(model_spec, "cuda", causal=True, allow_download=spec["allow_download"],
                             cpu_offload=True, gpu_memory_gib=spec["placement"], offload_folder=out / "offload" / str(spec["placement"]))
    sync()
    load_seconds = time.perf_counter() - started
    dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    if dtypes != ["torch.bfloat16"]:
        raise ValueError(f"expected native BF16 parameters, got {dtypes}")
    metadata = {"placement_gpu_gib": spec["placement"], "cpu_gib": 28, "load_seconds": load_seconds,
                "hf_device_map": {str(key): str(value) for key, value in getattr(model, "hf_device_map", {}).items()},
                "parameter_dtypes": dtypes, "setup_id": setup_id, "hardware": hardware(out, True)}
    sealed(Path(spec["attempt_dir"]) / "model-load.json", metadata)
    emit("model_loaded", metadata=metadata)
    stage, stage_deadline, reference = None, group_deadline, None
    stage_started, stage_consumed = time.monotonic(), {}
    reference_path = out / "reference-F.pt"
    reference_record = out / "reference-F.json"
    if reference_record.exists():
        saved = unseal(reference_record)
        if saved["study_id"] != spec["study_id"] or saved["setup_id"] != setup_id or file_hash(reference_path) != saved["sha256"]:
            raise ValueError("reference F checksum/fingerprint mismatch")
        reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    for definition in spec["cells"]:
        if stage != definition["stage"]:
            transition = time.monotonic()
            stage_seconds = stage_remaining(stage, definition["stage"], stage_started, stage_consumed,
                                            spec["stage_seconds"], transition)
            stage, stage_started = definition["stage"], transition
            stage_deadline = min(group_deadline, time.monotonic() + stage_seconds)
            if stage_seconds >= 5:
                emit("stage", stage=stage, seconds=max(0, stage_deadline - time.monotonic()))
        remaining = min(stage_deadline, group_deadline) - time.monotonic()
        if remaining < 5:
            publish_cell(spec, cell_payload(spec, definition, setup_id, status="skipped", reason="stage/group budget exhausted"))
            continue
        cap = min(remaining, 3600 if definition["mode"] == "FULL" else 900)
        emit("cell_start", name=definition["name"], seconds=cap)
        cell_deadline = min(stage_deadline, time.monotonic() + cap)
        artifact = Path(spec["attempt_dir"]) / (definition["name"] + ".h5")
        artifacts = {"input-selection.json": file_hash(out / "input-selection.json")}
        default_threads = torch.get_num_threads()
        try:
            torch.cuda.reset_peak_memory_stats()
            if definition["threads"]:
                torch.set_num_threads(definition["threads"])
            if definition["mode"] == "FULL":
                actual = full_run(model, tokenizer, picks[definition["percentile"]], model_spec, artifact,
                                  raw=definition["name"].endswith("raw") and not definition["name"].endswith("nonraw"))
                artifacts[artifact.relative_to(out).as_posix()] = file_hash(artifact)
            elif definition["mode"] == "AUX":
                if reference is None:
                    raise OverflowError("F reference unavailable; auxiliary cannot fabricate captured activations")
                actual = auxiliary(reference["datasets"], artifact)
                artifacts[artifact.relative_to(out).as_posix()] = file_hash(artifact)
            else:
                case = picks[definition["percentile"]]
                tape, tape_path = get_tape(spec, model, tokenizer, case, setup_id)
                artifacts[tape_path.relative_to(out).as_posix()] = file_hash(tape_path)
                original, question, rendered = c.render_prompt(tokenizer, case["text"])
                prefix, tokens = make_prefix(original, tape["ids"], definition["offset"], spec["decode_tokens"],
                                             model.config.max_position_embeddings)
                measurements, warmup, last = [], None, None
                repeat_skips = []
                for repeat in range(3):
                    if time.monotonic() >= cell_deadline - 2 or (warmup and cell_deadline - time.monotonic() <
                                                                1.15 * (warmup["prefill_seconds"] + warmup["decode_seconds"])):
                        repeat_skips.append({"repeat": repeat, "reason": "insufficient remaining cell budget"})
                        continue
                    if definition["mode"] in "ABCDE":
                        measured = staged_window(model, prefix, tokens, definition["mode"])
                    else:
                        measured = replay_window(model, tokenizer, case, prefix, question, rendered, tokens,
                                                 definition["mode"] == "G", model_spec)
                    datasets = measured.pop("datasets", None)
                    measured.pop("manifest", None)
                    if repeat == 0:
                        warmup = measured
                    else:
                        measurements.append(measured)
                        last = {"datasets": datasets, "measurement": measured}
                if not measurements:
                    raise OverflowError("warmup only/no completed measured repeat in budget")
                actual = {"case_id": case["case_id"], "prompt_tokens": int(original.numel()),
                          "prefix_tokens": int(prefix.numel()), "generated_offset": definition["offset"],
                          "tape_sha256": digest(tape["ids"]), "window_ids": tokens, "synthetic_load": tape["synthetic_load"],
                          "threads": torch.get_num_threads(), "warmup": warmup, "measurements": measurements,
                          "requested_repeats": 2, "completed_repeats": len(measurements), "skipped_repeats": repeat_skips,
                          "replay_scope": "exact production operations for this window; phase/selector history starts at window, prefix reconstructed by prefill"}
                if definition["name"] == "matrix_F":
                    reference = last
                    temporary = reference_path.with_suffix(".pt.tmp")
                    torch.save(reference, temporary)
                    temporary.replace(reference_path)
                    sealed(reference_record, {"study_id": spec["study_id"], "setup_id": setup_id,
                                               "sha256": file_hash(reference_path)})
                    artifacts[reference_path.name] = file_hash(reference_path)
                    artifacts[reference_record.name] = file_hash(reference_record)
                if definition["name"] == "candidate_F_threads1":
                    if reference is None:
                        raise OverflowError("baseline F absent, candidate equivalence cannot be established")
                    actual["equivalence"] = compare_datasets(reference["datasets"], last["datasets"])
                    for key in ("logit_endpoints_sha256", "argmax_endpoints"):
                        if reference["measurement"][key] != last["measurement"][key]:
                            raise ValueError("candidate forward logits/argmax differ")
                    actual["equivalence"]["logit_endpoints_exact"] = True
                    actual["speed_ratio_baseline_over_candidate"] = reference["measurement"]["decode_seconds"] / statistics.median(m["decode_seconds"] for m in measurements)
            actual["hardware_after"] = hardware(out, True)
            actual["configuration"] = metadata
            publish_cell(spec, cell_payload(spec, definition, setup_id, actual, artifacts=artifacts))
        except OverflowError as exc:
            publish_cell(spec, cell_payload(spec, definition, setup_id, status="skipped", reason=str(exc)))
        except BaseException as exc:
            publish_cell(spec, cell_payload(spec, definition, setup_id, status="failed",
                                            reason=f"{type(exc).__name__}: {exc}"))
            # Configuration is poisoned after OOM/nonfinite/numerical failures.
            raise
        finally:
            torch.set_num_threads(default_threads)
        emit("cell_end", name=definition["name"])


def fake_child(spec):
    """Protocol/cleanup test. Deliberately before every torch/model import."""
    if spec.get("spawn_descendant"):
        marker = str(Path(spec["attempt_dir"]) / "orphan-marker")
        code = "import time,pathlib; time.sleep(2); pathlib.Path(" + repr(marker) + ").write_text('orphan')"
        child = subprocess.Popen([sys.executable, "-c", code], creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        emit("descendant", pid=child.pid)
    emit("stage", stage="matrix", seconds=spec.get("fake_stage_seconds", 20))
    emit("cell_start", name=spec["cells"][0]["name"], seconds=spec.get("fake_cell_seconds", 20))
    if spec.get("fake_success"):
        definition = spec["cells"][0]
        publish_cell(spec, cell_payload(spec, definition, "fake", {"test": True}))
        return
    time.sleep(spec.get("delay", 10))


def child_main(path):
    spec = json.loads(path.read_text(encoding="utf-8"))
    try:
        if spec.get("fake"):
            fake_child(spec)
        else:
            child_group(spec)
        emit("group_done")
    except BaseException as exc:
        emit("group_failed", reason=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        return 1
    return 0


def stop_tree(process):
    """Only the process tree created by this parent; no name-wide termination."""
    if os.name == "nt":
        if process.poll() is None:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True,
                           creationflags=subprocess.CREATE_NO_WINDOW, timeout=15)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def supervise(spec, global_deadline, on_event, on_tick, interrupt_after=None):
    """Stdlib parent. Enforces min(cell,stage,group,global), even inside a stalled CUDA call."""
    config = Path(spec["attempt_dir"]) / "group-config.json"
    atomic(config, spec)
    messages = queue.Queue()
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--child", str(config)],
                               cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", bufsize=1,
                               creationflags=flags, start_new_session=os.name != "nt",
                               env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
    def reader():
        for line in process.stdout:
            messages.put(line)
        messages.put(None)
    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()
    start = time.monotonic()
    group_deadline = min(global_deadline, start + spec["group_seconds"])
    stage_deadline = cell_deadline = group_deadline
    heartbeat, eof, completed = start, False, False
    outcome = {"status": "failed", "reason": "child exited without group_done"}
    log = Path(spec["attempt_dir"]) / "child.log"
    try:
        with log.open("w", encoding="utf-8") as stream:
            while not eof or process.poll() is None:
                current = time.monotonic()
                if interrupt_after is not None and current - start >= interrupt_after:
                    raise KeyboardInterrupt("synthetic parent interrupt")
                if current >= min(group_deadline, stage_deadline, cell_deadline):
                    outcome = {"status": "timeout", "reason": "parent enforced minimum cell/stage/group/global deadline"}
                    stop_tree(process)
                    break
                if current >= heartbeat:
                    on_tick(current - start)
                    heartbeat = current + 5
                try:
                    line = messages.get(timeout=.1)
                except queue.Empty:
                    continue
                if line is None:
                    eof = True
                    continue
                stream.write(line)
                stream.flush()
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = event.get("diagnostic_event")
                if kind == "stage":
                    stage_deadline = min(group_deadline, time.monotonic() + max(0, event["seconds"]))
                    cell_deadline = group_deadline
                elif kind == "cell_start":
                    cell_deadline = min(stage_deadline, group_deadline, time.monotonic() + max(0, event["seconds"]))
                elif kind in ("cell_end", "cell_result"):
                    cell_deadline = group_deadline
                elif kind == "group_done":
                    completed = True
                    outcome = {"status": "success"}
                elif kind == "group_failed":
                    outcome = {"status": "failed", "reason": event["reason"]}
                if kind:
                    on_event(event, time.monotonic() - start)
    except KeyboardInterrupt:
        stop_tree(process)
        on_tick(time.monotonic() - start)
        raise
    finally:
        if process.poll() is None:
            stop_tree(process)
        process.stdout.close()
        reader_thread.join(timeout=2)
        on_tick(time.monotonic() - start)
    if process.returncode and completed:
        outcome = {"status": "failed", "reason": f"child exit code {process.returncode} after group_done"}
    return outcome


def verify_row(out, row, study_id):
    if row["status"] != "success":
        return False
    path = out / row["relative_result"]
    if file_hash(path) != row["result_sha256"]:
        raise ValueError(f"completed result checksum changed: {path}")
    payload = unseal(path)
    if payload["study_id"] != study_id or payload["fingerprint"] != digest(payload["signature"]):
        raise ValueError("completed result fingerprint differs")
    for key in ("name", "mode", "placement", "offset", "percentile", "threads", "actual"):
        if row.get(key) != payload.get(key):
            raise ValueError(f"state/result data differ: {key}")
    for relative, expected in payload["artifacts"].items():
        artifact = (out / relative).resolve()
        if not artifact.is_relative_to(out.resolve()) or file_hash(artifact) != expected:
            raise ValueError(f"artifact checksum/path mismatch: {relative}")
    return True


def restore_budget(state):
    """Charge a crashed parent's still-active allocation; never reset spent budget."""
    active = state.pop("active", None)
    if active:
        elapsed = min(active["allocation"], max(0, time.time() - active["started_unix"]))
        state["spent_seconds"] = max(state["spent_seconds"], active["spent_before"] + elapsed)
        key = str(active["placement"])
        state["group_spent"][key] = max(state["group_spent"].get(key, 0), active["group_before"] + elapsed)
        state["recovered_unclean_parent"] = True


def validate_out(out):
    out = out.resolve()
    forbidden = ("split.json", "study.json", "result.json", "collection-status.json")
    if out.exists() and not ((out / "diagnostic-state.json").exists() or (out / PREFLIGHT_MARKER).exists()):
        raise ValueError("--out already exists and is not this diagnostic: refusing to write")
    if any((out / name).exists() for name in forbidden) or (out / "cases").exists():
        raise ValueError("--out looks like a study/production archive: refusing to write")
    if any((parent / "collection-status.json").exists() or (parent / "split.json").exists()
           for parent in [out, *out.parents]):
        raise ValueError("--out is nested inside a study/collection directory")
    return out


def main(args):
    if args.self_test:
        return self_test()
    if not args.out:
        raise ValueError("--out is required")
    if not 0 < args.hours <= 24 or not 1 <= args.decode_tokens <= 64:
        raise ValueError("require 0<hours<=24 and 1<=decode-tokens<=64")
    out = validate_out(args.out)
    identity = study_identity(args)
    study_id = digest(identity)
    state_path = out / "diagnostic-state.json"
    if state_path.exists():
        state = unseal(state_path)
        if state["study_id"] != study_id or state["identity"] != identity:
            raise ValueError("scientific/source/runtime/input fingerprint changed; choose a new --out")
        if state["budget_seconds"] != args.hours * 3600:
            raise ValueError("resume must keep original --hours budget; it cannot be enlarged/reset")
        restore_budget(state)
        for row in state["cells"]:
            verify_row(out, row, study_id)
    else:
        state = None
    verify_preflight_marker(out, identity, study_id)
    try:
        model_spec, setup = preflight_inputs(args, out)
    except Exception as exc:
        write_preflight_failure(out, exc)
        return 2
    print(canonical({"preflight": "passed", "inputs": len(setup["token_inventory"]),
                     "model": model_spec["model"], "revision": model_spec["revision"]}), flush=True)
    if args.preflight_only:
        write_preflight_marker(out, identity, study_id)
        print(f"Preflight report: {out / 'input-selection.json'}", flush=True)
        return 0
    if state is None:
        out.mkdir(parents=True, exist_ok=True)
        state = {"schema": SCHEMA, "study_id": study_id, "identity": identity,
                 "budget_seconds": args.hours * 3600, "spent_seconds": 0, "group_spent": {}, "stage_spent": {},
                 "hardware_initial": hardware(out), "cells": [{**definition, "status": "pending"} for definition in make_cells()],
                 "attempts": []}
    def save():
        sealed(state_path, state)
        report(out, state)
    save()
    session_started, session_spent_before = time.monotonic(), state["spent_seconds"]
    global_deadline = time.monotonic() + max(0, state["budget_seconds"] - state["spent_seconds"])
    scale = state["budget_seconds"] / 18000
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt("parent termination signal")
    old_term = signal.signal(signal.SIGTERM, interrupted)
    try:
        for placement, fraction in ((14, 13 / 15), (13, 2 / 15)):
            definitions = [d for d in make_cells() if d["placement"] == placement and
                           next(r for r in state["cells"] if r["name"] == d["name"])["status"] != "success"]
            if not definitions:
                continue
            remaining = min(global_deadline - time.monotonic(), state["budget_seconds"] * fraction - state["group_spent"].get(str(placement), 0))
            if remaining < 5:
                for row in state["cells"]:
                    if row["placement"] == placement and row["status"] != "success":
                        row.update(status="skipped", reason="persisted group/global budget exhausted")
                save()
                continue
            attempt_dir = out / "attempts" / f"gpu{placement}-{time.time_ns()}"
            attempt_dir.mkdir(parents=True)
            limits = {"full": 7200, "matrix": 7200, "aux": 1200, "placement": 2400}
            spec = {"study_id": study_id, "out": str(out), "attempt_dir": str(attempt_dir), "placement": placement,
                    "cells": definitions, "group_seconds": remaining,
                    "stage_seconds": {key: max(0, value * scale - state["stage_spent"].get(f"{placement}:{key}", 0))
                                      for key, value in limits.items()},
                    "originals": str(args.originals.resolve()), "variations": str(args.variations.resolve()),
                    "allow_download": args.allow_download, "decode_tokens": args.decode_tokens,
                    "synthetic_repeat": args.synthetic_repeat}
            spent_before = state["spent_seconds"]
            group_before = state["group_spent"].get(str(placement), 0)
            stage_info = {"name": "setup", "start": 0.0, "before": state["stage_spent"].get(f"{placement}:setup", 0)}
            current_cell = [None]
            state["active"] = {"started_unix": time.time(), "allocation": remaining, "spent_before": spent_before,
                               "placement": placement, "group_before": group_before}
            # Immediately invalidate failed/pending retries. Old successes are never retried.
            for row in state["cells"]:
                if row["name"] in {d["name"] for d in definitions}:
                    name = row["name"]
                    row.clear()
                    row.update(next(d for d in definitions if d["name"] == name), status="pending")
            save()
            def account(elapsed):
                state["spent_seconds"] = spent_before + elapsed
                state["group_spent"][str(placement)] = group_before + elapsed
                state["stage_spent"][f"{placement}:{stage_info['name']}"] = stage_info["before"] + max(0, elapsed - stage_info["start"])
            def on_tick(elapsed):
                account(elapsed)
                save()
                print(canonical({"heartbeat": True, "gpu_gib": placement, "cell": current_cell[0],
                                 "stage": stage_info["name"], "spent_seconds": round(state["spent_seconds"], 1)}), flush=True)
            def on_event(event, elapsed):
                account(elapsed)
                kind = event["diagnostic_event"]
                if kind == "stage":
                    name = event["stage"]
                    stage_info.update(name=name, start=elapsed, before=state["stage_spent"].get(f"{placement}:{name}", 0))
                elif kind == "cell_start":
                    current_cell[0] = event["name"]
                    next(r for r in state["cells"] if r["name"] == event["name"])["status"] = "running"
                elif kind == "cell_result":
                    relative = event["relative_result"]
                    result_path = (out / relative).resolve()
                    if not result_path.is_relative_to(attempt_dir.resolve()):
                        raise ValueError("child result outside owned attempt")
                    payload = unseal(result_path)
                    if payload["study_id"] != study_id:
                        raise ValueError("child result study identity differs")
                    target = next(r for r in state["cells"] if r["name"] == payload["name"])
                    target.clear()
                    target.update(payload, relative_result=relative, result_sha256=file_hash(result_path))
                    verify_row(out, target, study_id)
                    current_cell[0] = None
                elif kind == "model_loaded":
                    state.setdefault("model_loads", []).append(event["metadata"])
                save()
            try:
                outcome = supervise(spec, global_deadline, on_event, on_tick)
            except KeyboardInterrupt:
                outcome = {"status": "interrupted", "reason": "parent Ctrl+C/SIGTERM; owned tree stopped"}
                state["stopped"] = "interrupted"
            except Exception as exc:
                outcome = {"status": "failed", "reason": f"parent validation/error: {type(exc).__name__}: {exc}"}
            finally:
                state.pop("active", None)
            state["attempts"].append({"placement": placement, "directory": attempt_dir.relative_to(out).as_posix(), **outcome})
            for row in state["cells"]:
                if row["placement"] == placement and row["status"] in ("running", "pending"):
                    row.update(status="skipped", reason=f"configuration stopped: {outcome.get('reason', outcome['status'])}")
            save()
            if outcome["status"] == "interrupted":
                break
        for row in state["cells"]:
            if row["status"] == "pending":
                row.update(status="skipped", reason="not run after interruption/budget exhaustion")
        state["spent_seconds"] = max(state["spent_seconds"], session_spent_before + time.monotonic() - session_started)
        save()
    finally:
        signal.signal(signal.SIGTERM, old_term)
    print(f"Report: {out / 'summary.md'}", flush=True)
    return 0


def self_test():
    """Real tiny-Qwen CPU checks + fake subprocesses. Never calls model_context."""
    import torch
    from research import _collector as c, _support as io
    from research._activation_store import geometry_channels
    from transformers import Qwen3Config, Qwen3ForCausalLM
    tests = []
    def check(condition, name):
        if not condition:
            raise AssertionError(name)
        tests.append(name)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("self-test must not access cached tokenizer/model weights")
    default_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    tiny = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                                       num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                       max_position_embeddings=4096)).to(dtype=torch.bfloat16).eval()
    class Tokenizer:
        eos_token_id = None
        chat_template = "tiny-test-only"
        def __call__(self, text, **_kwargs):
            return {"input_ids": [7, 8] if text == "</think>" else [4]}
        def decode(self, ids, **_kwargs):
            return " ".join(map(str, ids))
    tokenizer = Tokenizer()
    case = {"case_id": "tiny:original", "family_id": "tiny", "variant_id": "original", "text": "question"}
    prompt = torch.tensor([1, 2, 3])
    tokens = [4, 7, 8, 5, 6]
    try:
        with patched(io, model_context=forbidden, load_backbone=forbidden), patched(c, LAYERS=(0, 1)):
            class PromptTokenizer:
                def __init__(self, template): self.template = template
                def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                    text = "U:" + messages[0]["content"] + "\\nA:"
                    return self.template if tokenize else text
                def __call__(self, text, **_kwargs):
                    return {"input_ids": [1, 2, 3, 4, 5, 6],
                            "offset_mapping": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]}
            expected_ids = [1, 2, 3, 4, 5, 6]
            for template in (expected_ids, {"input_ids": expected_ids, "attention_mask": [1] * len(expected_ids)}):
                rendered_prompt, positions, rendered_text = c.render_prompt(PromptTokenizer(template), "q")
                check(rendered_prompt.tolist() == expected_ids and positions == [2] and rendered_text == "U:q\\nA:",
                      "render_prompt accepts " + ("BatchEncoding" if isinstance(template, dict) else "token-ID list"))
            try:
                c.render_prompt(PromptTokenizer([1, 2, 9, 4, 5, 6]), "q")
            except ValueError as exc:
                check("differ" in str(exc), "render_prompt rejects genuine unequal token IDs")
            else:
                raise AssertionError("unequal template IDs accepted")
            prefix, window = make_prefix(prompt, list(range(20)), 8, 5, 4096)
            check(prefix.tolist() == [1, 2, 3] + list(range(8)) and window == list(range(8, 13)), "G offset appends tape, preserves original prompt")
            try:
                make_prefix(prompt, list(range(20)), 8, 5, 10)
            except OverflowError:
                tests.append("context overflow is explicit skip")
            else:
                raise AssertionError("context overflow was accepted")
            results = {mode: staged_window(tiny, prefix, tokens, mode) for mode in "ABCDE"}
            check(len({r["logit_endpoints_sha256"] for r in results.values()}) == 1, "cached Qwen A-E logits identical")
            check(len({tuple(r["argmax_endpoints"]) for r in results.values()}) == 1, "cached Qwen A-E argmax identical")
            check(results["A"]["counters"]["take"] == 0 and results["B"]["counters"]["take"] == 5, "B actual hooks distinct from A")
            check(results["C"]["counters"]["geometry"] == 10 and results["C"]["counters"]["welford"] == 0, "C geometry without Welford")
            check(results["D"]["counters"]["output_row"] == 5 and results["C"]["counters"]["output_row"] == 0, "D actual output row added")
            check(results["E"]["counters"]["native_copies"] == 30 and results["E"]["counters"]["geometry"] == 10, "E native copies without duplicate geometry")
            f = replay_window(tiny, tokenizer, case, prefix, [1], "question", tokens, False, {})
            g = replay_window(tiny, tokenizer, case, prefix, [1], "question", tokens, True, {})
            check(f["logit_endpoints_sha256"] == results["A"]["logit_endpoints_sha256"], "exact production replay preserves full logit endpoints")
            check(f["datasets"]["generated_ids"].tolist() == tokens, "replay teacher forces identical IDs")
            check(f["datasets"]["service_mask"].tolist() == [0, 1, 1, 0, 0], "actual pending close-marker service handling")
            check("raw_g_h_bf16" in g["datasets"] and "raw_g_h_bf16" not in f["datasets"], "G actual raw packing distinct from F")
            check(f["datasets"]["phase_h_n"].tolist() == [1, 1, 2, 0], "actual Q/R/A h Welford counts")
            torch.set_num_threads(1)
            candidate = replay_window(tiny, tokenizer, case, prefix, [1], "question", tokens, False, {})
            comparison = compare_datasets(f["datasets"], candidate["datasets"])
            check(comparison["passed"] and f["argmax_endpoints"] == candidate["argmax_endpoints"], "candidate all datasets numerical equivalence")
            torch.set_num_threads(2)
            with torch.inference_mode(), c.HookCapture(tiny) as hooks:
                tiny(input_ids=prompt.unsqueeze(0), use_cache=True)
                capture = hooks.take()
            vectors = capture[0]
            actual = geometry_channels(*(vectors[name][0, 0] for name in ("r", "a", "m", "u", "h")))
            r, a, m, u, h = (vectors[name][0, 0].double() for name in ("r", "a", "m", "u", "h"))
            dot = lambda x, y: (x * y).sum()
            expected = torch.stack([dot(r,r), dot(a,a), dot(m,m), dot(r,a), dot(r,m), dot(a,m), dot(a+m,a+m),
                                    dot(h,h), dot(u,u), dot(r,u), dot(u,h), dot(r,h), dot(h-r,h-r)])
            check(torch.allclose(actual.double(), expected, atol=2e-5, rtol=2e-5), "all 13 actual hook geometry channels against FP64")
            w = paired_welford(torch.cat([v["h"][0] for v in capture.values()], dim=1))
            check(all(row["paired_exact"] and row["fp64_reference_passed"] for row in w), "captured CPU h Welford threads default/1/4 FP64")
            with tempfile.TemporaryDirectory(prefix="aimo-diag-selftest-") as directory:
                out = Path(directory)
                aux = auxiliary(f["datasets"], out / "aux.h5")
                check(aux["file_bytes"] > 0 and aux["write_checksum_readback_seconds"] > 0, "actual packed HDF5/checksum/readback")
                with torch.inference_mode():
                    tokenizer.eos_token_id = int(tiny(input_ids=prompt.unsqueeze(0)).logits[0, -1].argmax())
                with patched(c, render_prompt=lambda *_: (prompt, [1], "question")):
                    full = full_run(tiny, tokenizer, case, {}, out / "full.h5", False)
                check(full["generated_tokens"] == 1 and full["stop_reason"] == "eos", "FULL natural EOS honored and actual G read from verified HDF5")
                rows = [{**cell("ablation", "A"), "status": "success", "actual": {"decode_tok_s": 99999}},
                        {**cell("short", "FULL"), "status": "success", "actual": full}]
                check(len(compatible_full_eta(rows)) == 1 and compatible_full_eta(rows)[0]["conditional_days_all2048"] is None,
                      "ETA excludes ablations and rejects EOS-short 2048 extrapolation")
                rows[1]["actual"] = {**full, "generated_tokens": 2048, "stop_reason": "max_new_tokens", "end_to_end_seconds": 300}
                check(compatible_full_eta(rows)[0]["conditional_days_all2048"] == 1507 * 300 / 86400, "conditional ETA uses FULL end-to-end only")
                definition = cell("fake", "A")
                spec = {"study_id": "test", "out": str(out), "cells": [definition], "fake": True,
                        "placement": 14, "group_seconds": 10}
                events = []
                def event(value, elapsed):
                    events.append(value)
                for variant in ("timeout_cell", "timeout_global", "interrupt", "success"):
                    attempt = out / variant
                    attempt.mkdir()
                    trial = {**spec, "attempt_dir": str(attempt), "spawn_descendant": variant != "success",
                             "fake_cell_seconds": .5 if variant == "timeout_cell" else 20, "fake_success": variant == "success"}
                    started = time.monotonic()
                    try:
                        result = supervise(trial, started + (.5 if variant == "timeout_global" else 10),
                                           event, lambda _: None, interrupt_after=.5 if variant == "interrupt" else None)
                    except KeyboardInterrupt:
                        check(variant == "interrupt", "parent interrupt caught and child tree stopped")
                    else:
                        check(result["status"] == ("success" if variant == "success" else "timeout"), variant + " correct status")
                    check(time.monotonic() - started < 4, variant + " bounded parent time")
                time.sleep(2.2)  # Descendants would have written after 2s if any survived.
                check(not list(out.glob("*/orphan-marker")), "timeout/interrupt leave no owned descendant")
                result_event = next(e for e in reversed(events) if e["diagnostic_event"] == "cell_result")
                result_path = out / result_event["relative_result"]
                payload = unseal(result_path)
                row = {**payload, "relative_result": result_event["relative_result"], "result_sha256": file_hash(result_path)}
                check(verify_row(out, row, "test"), "resume retains verified actual data")
                try:
                    verify_row(out, row, "changed")
                except ValueError:
                    tests.append("resume changed fingerprint rejected")
                else:
                    raise AssertionError("changed fingerprint accepted")
                result_path.write_text("corrupt", encoding="utf-8")
                try:
                    verify_row(out, row, "test")
                except ValueError:
                    tests.append("resume corrupt completed result rejected")
                else:
                    raise AssertionError("corrupt completed result accepted")
                state = {"spent_seconds": 4, "group_spent": {"14": 4},
                         "active": {"allocation": 5, "started_unix": time.time()-100, "spent_before": 3,
                                    "group_before": 3, "placement": 14}}
                restore_budget(state)
                check(state["spent_seconds"] == 8 and state["group_spent"]["14"] == 8 and "active" not in state,
                      "resume charges previous active budget without reset")
                consumed, limits = {}, {"matrix": 60, "full": 120}
                check(stage_remaining(None, "matrix", 0, consumed, limits, 10) == 60, "matrix initial budget")
                stage_remaining("matrix", "full", 10, consumed, limits, 30)
                remaining = stage_remaining("full", "matrix", 30, consumed, limits, 50)
                check(remaining == 40 and consumed == {"matrix": 20, "full": 20},
                      "return from FULL retains spent core-matrix budget")
                schedule = [row["name"] for row in make_cells()]
                check(schedule.index("candidate_F_threads1") < schedule.index("full_P50_nonraw") < schedule.index("window_P10_G0_A"),
                      "core bottleneck and candidate precede potentially timing-out FULL")
                try:
                    validate_out(out)
                except ValueError:
                    tests.append("unowned output directory refused")
                else:
                    raise AssertionError("unowned output accepted")
                module = sys.modules[__name__]
                identity = {"lifecycle": "stable"}
                def fake_preflight(args, target):
                    sealed(target / "input-selection.json", {"token_inventory": [{"case_id": "fake", "tokens": 1}]})
                    return {"model": "fake", "revision": "pinned"}, {"token_inventory": [{"case_id": "fake", "tokens": 1}]}
                def fake_supervise(*_args, **_kwargs):
                    return {"status": "success", "reason": "fake child"}
                def lifecycle_args(target, preflight_only):
                    return SimpleNamespace(self_test=False, out=target, hours=1, decode_tokens=1,
                                           preflight_only=preflight_only, allow_download=False,
                                           synthetic_repeat=True, originals=target / "originals.json",
                                           variations=target / "variations.json")
                with patched(module, study_identity=lambda _args: identity, preflight_inputs=fake_preflight,
                             make_cells=lambda: [cell("fake", "A")], supervise=fake_supervise):
                    fresh = out / "normal-fresh"
                    check(main(lifecycle_args(fresh, False)) == 0 and (fresh / "diagnostic-state.json").exists(),
                          "normal fresh start creates state after preflight input selection")
                    staged = out / "preflight-then-normal"
                    check(main(lifecycle_args(staged, True)) == 0 and (staged / PREFLIGHT_MARKER).exists()
                          and not (staged / "diagnostic-state.json").exists(),
                          "preflight-only writes verified diagnostic marker")
                    check(main(lifecycle_args(staged, False)) == 0 and (staged / "diagnostic-state.json").exists(),
                          "normal run accepts same verified preflight output")
                    before = {name: file_hash(staged / name) for name in (PREFLIGHT_MARKER, "input-selection.json", "diagnostic-state.json")}
                    with patched(module, study_identity=lambda _args: {"lifecycle": "changed"}):
                        try:
                            main(lifecycle_args(staged, True))
                        except ValueError:
                            tests.append("mismatched resume rejects before preflight mutation")
                        else:
                            raise AssertionError("mismatched resume accepted")
                    check(before == {name: file_hash(staged / name) for name in before},
                          "mismatched resume leaves diagnostic artifacts unchanged")
    finally:
        torch.set_num_threads(default_threads)
    print(canonical({"self_test": "passed", "checks": len(tests), "tests": tests,
                     "limits": "CPU tiny random Qwen only; no remote checkpoint, CUDA, 16GB placement or real 5h performance tested"}))
    return 0


if __name__ == "__main__":
    # Windows redirected consoles otherwise may encode Russian help in cp1251.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    arguments = parser().parse_args()
    if arguments.child:
        raise SystemExit(child_main(arguments.child))
    raise SystemExit(main(arguments))
