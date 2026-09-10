"""Служебный код: локальные данные, кеш, журнал, CLI и переносимый solution ZIP.
Не ML-framework. Научные операции живут в features/predictor/experiment.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import time
import traceback
from types import SimpleNamespace
from zipfile import ZipFile, ZIP_DEFLATED
import torch

ROOT = Path(__file__).resolve().parents[1]
ALIASES = {"qwen3-8b:low": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def read_rows(path: Path, model: str | None = None) -> list[dict]:
    if path.suffix == ".parquet":
        import pandas as pd
        rows = json.loads(pd.read_parquet(path).to_json(orient="records"))
    else:
        text = path.read_text(encoding="utf-8")
        rows = json.loads(text) if path.suffix == ".json" else [json.loads(s) for s in text.splitlines() if s.strip()]
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise ValueError("Expected JSON records or JSONL/Parquet rows")
    result, ids, seen_texts = [], set(), {}
    for i, r in enumerate(rows):
        row_model = r.get("model_id")
        if row_model and model and ALIASES.get(row_model, row_model) != ALIASES.get(model, model):
            raise ValueError("Mixed/wrong model_id: explicitly filter source rows before preparing a study")
        problem = r.get("original_problem", r.get("problem"))
        label = r.get("model_is_robust", r.get("is_robust"))
        family = r.get("family_id")
        if family is None and r.get("dataset_id") is not None and r.get("problem_id") is not None:
            family = canonical([str(r["dataset_id"]), str(r["problem_id"])])
        if not isinstance(problem, str) or not problem.strip() or type(label) is not bool:
            raise ValueError(f"Row {i}: require problem string and real boolean robustness label")
        if not isinstance(family, str) or not family:
            raise ValueError(f"Row {i}: supply family_id or dataset_id + problem_id, never infer it from wording")
        identity = r.get("id") or digest(canonical([family, problem, i]).encode())
        if not isinstance(identity, str) or identity in ids:
            raise ValueError("IDs must be unique strings")
        if problem in seen_texts and seen_texts[problem] != (family, label):
            raise ValueError("Same input has conflicting family/label; resolve explicitly, not by dropping rows")
        seen_texts[problem] = (family, label)
        ids.add(identity)
        result.append(dict(id=identity, family_id=family, problem=problem, is_robust=label, model_id=row_model))
    if not result:
        raise ValueError("Empty data")
    models = {ALIASES.get(r["model_id"], r["model_id"]) for r in result if r["model_id"]}
    if len(models) > 1:
        raise ValueError("Prepare a separate study per evaluated LLM")
    return result


def load_splits(train_path: Path, valid_path: Path, model: str):
    train, valid = read_rows(train_path, model), read_rows(valid_path, model)
    for key in ("id", "family_id", "problem"):
        if {r[key] for r in train} & {r[key] for r in valid}:
            raise ValueError(f"Train/validation overlap in {key}")
    if {r["is_robust"] for r in train} != {False, True}:
        raise ValueError("Training requires both classes")
    return train, valid


def prepare(args) -> None:
    rows = read_rows(args.source, args.model)
    groups = sorted({r["family_id"] for r in rows})
    if len(groups) < 2 or not 0 < args.validation_fraction < 1:
        raise ValueError("Need at least two independent families and a fraction in (0,1)")
    random.Random(args.seed).shuffle(groups)
    size = min(len(groups) - 1, max(1, round(len(groups) * args.validation_fraction)))
    selected = set(groups[:size])
    train = [r for r in rows if r["family_id"] not in selected]
    valid = [r for r in rows if r["family_id"] in selected]
    if {r["is_robust"] for r in train} != {False, True}:
        raise ValueError("Proposed train lacks a class; design a grouped split explicitly, not by looking at scores")
    args.out.mkdir(parents=True, exist_ok=False)
    for name, subset in (("train", train), ("validation", valid)):
        (args.out / f"{name}.jsonl").write_text("".join(canonical(r) + "\n" for r in subset), encoding="utf-8")
    write_json(args.out / "split.json", dict(source_sha256=file_hash(args.source), seed=args.seed,
               validation_fraction=args.validation_fraction, train_families=sorted(set(groups) - selected),
               validation_families=sorted(selected)))
    print(f"Prepared {len(train)} train / {len(valid)} validation rows in {args.out}")


def check_deadline(args) -> None:
    if time.monotonic() >= args._deadline:
        raise TimeoutError("Cooperative time budget reached between batches/epochs")


def model_context(args):
    from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerFast, __version__
    identifier = ALIASES.get(args.model, args.model)
    kw = dict(revision=args.revision, local_files_only=not getattr(args, "allow_download", False), trust_remote_code=False)
    config = AutoConfig.from_pretrained(identifier, **kw)
    if getattr(config, "_commit_hash", None):
        kw["revision"] = config._commit_hash  # pin tokenizer to the same snapshot
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(identifier, **kw)
    except (OSError, TypeError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(identifier, **kw)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs a pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    local = Path(identifier)
    revision = getattr(config, "_commit_hash", None)
    if local.is_dir():
        identifier = str(local.resolve())
        # Exact local checkpoint identity, including shards and tokenizer assets.
        revision = digest(canonical({p.relative_to(local).as_posix(): file_hash(p)
            for p in sorted(local.rglob("*")) if p.is_file() and ".git" not in p.parts
            and p.suffix in {".json", ".safetensors", ".bin", ".model", ".txt"}}).encode())
    elif not revision:
        raise ValueError("Could not resolve cached model revision")
    tokenizer_state = dict(vocab=tokenizer.get_vocab(), special=tokenizer.special_tokens_map,
                           template=tokenizer.chat_template)
    spec = dict(model=identifier, revision=revision, local=local.is_dir(), layers=getattr(args, "layers", None),
                max_length=getattr(args, "max_length", None), template=getattr(args, "template", "chat"), padding="right", truncation="right",
                dtype=args.dtype, device=args.device, transformers=__version__, torch=str(torch.__version__),
                tokenizer_sha256=digest(canonical(tokenizer_state).encode()),
                extractor_sha256=file_hash(Path(__file__).with_name("features.py")))
    return tokenizer, spec


def load_backbone(spec, device, *, causal: bool = False, allow_download: bool = False, cpu_offload: bool = False,
                  gpu_memory_gib: int | None = None, offload_folder: Path | None = None):
    """Load exactly the recorded checkpoint; collection needs the causal-LM head."""
    from transformers import AutoModel, AutoModelForCausalLM
    cls = AutoModelForCausalLM if causal else AutoModel
    kw = dict(revision=None if spec["local"] else spec["revision"], local_files_only=not allow_download,
              trust_remote_code=False, dtype=getattr(torch, spec["dtype"]))
    if cpu_offload:
        if device != "cuda":
            raise ValueError("--cpu-offload requires --device cuda")
        if not gpu_memory_gib or gpu_memory_gib < 1 or offload_folder is None:
            raise ValueError("--cpu-offload requires positive --gpu-memory-gib and --offload-folder")
        offload_folder.mkdir(parents=True, exist_ok=True)
        kw.update(device_map="auto", max_memory={0: f"{gpu_memory_gib}GiB", "cpu": "28GiB"}, offload_folder=str(offload_folder))
        return cls.from_pretrained(spec["model"], **kw).eval()
    model = cls.from_pretrained(spec["model"], **kw)
    return model.to(device).eval()


def features(args, texts, extractor):
    check_deadline(args)
    tokenizer, spec = model_context(args)
    signature = digest(canonical(dict(spec=spec, texts=texts, batch_size=args.batch_size)).encode())
    cache = ROOT / ".runtime/features" / f"{signature}.pt"
    if cache.is_file():
        saved = torch.load(cache, map_location="cpu", weights_only=True)
        x = saved["x"]
        if saved["signature"] != signature or x.ndim != 3 or len(x) != len(texts) or x.shape[1] != len(args.layers) or not torch.isfinite(x).all():
            raise ValueError("Invalid feature cache")
        return x, spec
    check_deadline(args)
    llm = load_backbone(spec, args.device)
    try:
        batches = []
        for start in range(0, len(texts), args.batch_size):
            check_deadline(args)
            batches.append(extractor(llm, tokenizer, texts[start:start + args.batch_size], layers=args.layers,
                                     max_length=args.max_length, template=args.template))
        x = torch.cat(batches)
    finally:
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".tmp")
    torch.save(dict(signature=signature, x=x), temporary)
    temporary.replace(cache)
    return x, spec


def save_outputs(args, head, mean, scale, spec, valid, probabilities):
    directory = ROOT / ".runtime" / args.name
    payload = dict(schema=1, state_dict=head.state_dict(), mean=mean, scale=scale,
                   input_dim=mean.numel(), width=args.width, threshold=args.threshold,
                   batch_size=args.batch_size, feature_spec=spec)
    torch.save(payload, directory / "head.pt")
    predictions = [dict(id=r["id"], family_id=r["family_id"], label=r["is_robust"], probability=float(p))
                   for r, p in zip(valid, probabilities)]
    (directory / "predictions.jsonl").write_text("".join(canonical(r) + "\n" for r in predictions), encoding="utf-8")


def predict_bundle(directory: Path, model_id: str, problems: list[str]) -> list[bool]:
    from research.features import extract_features
    from research.predictor import build_predictor, predict
    if not problems:
        return []
    payload = torch.load(directory / "head.pt", map_location="cpu", weights_only=True)
    spec = payload["feature_spec"]
    if ALIASES.get(model_id, model_id) != spec["model"]:
        raise ValueError("This trained head belongs to a different LLM")
    head = build_predictor(payload["input_dim"], payload["width"])
    head.load_state_dict(payload["state_dict"])
    # Use exactly the saved prompt/feature settings; device must be available, no silent fallback.
    llm = load_backbone(spec, spec["device"])
    try:
        args = SimpleNamespace(model=spec["model"], revision=spec["revision"], layers=spec["layers"],
                 max_length=spec["max_length"], template=spec["template"], dtype=spec["dtype"], device=spec["device"])
        tokenizer, current = model_context(args)
        if current != spec:
            raise ValueError("Checkpoint/tokenizer/runtime differs from training feature spec")
        decisions = []
        for start in range(0, len(problems), payload["batch_size"]):
            x = extract_features(llm, tokenizer, problems[start:start + payload["batch_size"]],
                                 layers=spec["layers"], max_length=spec["max_length"], template=spec["template"])
            decisions.extend((predict(head, x, payload["mean"], payload["scale"]) >= payload["threshold"]).tolist())
        return decisions
    finally:
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def export(args):
    run_dir = ROOT / ".runtime" / args.name
    record = json.loads((ROOT / "experiments" / args.name / "result.json").read_text())
    if record["status"] != "completed" or file_hash(run_dir / "head.pt") != record["artifacts"]["head.pt"]:
        raise ValueError("Need a completed, unchanged training artifact")
    entry = ('from pathlib import Path\nfrom research._support import predict_bundle\n\n'
             'def are_robust(model_id: str, problems: list[str]) -> list[bool]:\n'
             '    return predict_bundle(Path(__file__).parent, model_id, problems)\n')
    for name, expected in record["source_files"].items():
        if Path(name).name != name or not name.endswith(".py"):
            raise ValueError("Invalid source snapshot path")
        if file_hash(run_dir / "source" / name) != expected:
            raise ValueError("Run source snapshot changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    owned = False
    try:
        with ZipFile(args.output, "x", compression=ZIP_DEFLATED) as bundle:
            owned = True
            bundle.writestr("solution.py", entry)
            bundle.write(run_dir / "head.pt", "head.pt")
            for name in record["source_files"]:
                bundle.write(run_dir / "source" / name, f"research/{name}")
    except BaseException:
        if owned:
            args.output.unlink(missing_ok=True)
        raise
    print(args.output)


def git(*args):
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), *args], stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@contextmanager
def journal(args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.name):
        raise ValueError("Use a simple unique run name")
    directory, raw = ROOT / "experiments" / args.name, ROOT / ".runtime" / args.name
    if directory.exists() or raw.exists():
        raise FileExistsError("Run name exists; never overwrite previous attempts")
    # One cooperative writer/GPU job per Git common-dir (also shared by worktrees).
    common = git("rev-parse", "--git-common-dir")
    lock = (ROOT / common if common else ROOT / ".runtime") / "aimo.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("x", encoding="utf-8") as f:
        f.write(args.name)
    try:
        directory.mkdir(parents=True)
        (raw / "source").mkdir(parents=True)
        sources = {}
        for path in sorted(Path(__file__).parent.glob("*.py")):
            (raw / "source" / path.name).write_bytes(path.read_bytes())
            sources[path.name] = file_hash(path)
        record = dict(status="running", scientific_status="unreviewed", name=args.name,
                      started_at=datetime.now(timezone.utc).isoformat(), code_sha=git("rev-parse", "HEAD"),
                      dirty=git("status", "--porcelain") != "", source_files=sources,
                      settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if not k.startswith("_")})
        write_json(directory / "result.json", record)
        args._deadline = time.monotonic() + args.max_seconds
        try:
            record["data_sha256"] = {key: file_hash(getattr(args, key)) for key in ("train", "validation")}
            yield record
            check_deadline(args)
            if any(file_hash(Path(__file__).with_name(name)) != sha for name, sha in sources.items()):
                raise RuntimeError("Source changed while experiment was running")
            if any(file_hash(getattr(args, key)) != sha for key, sha in record["data_sha256"].items()):
                raise RuntimeError("Input data changed while experiment was running")
            record["artifacts"] = {name: file_hash(raw / name) for name in ("head.pt", "predictions.jsonl")}
            record["status"] = "completed"
        except BaseException as exc:
            record["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            record["error_type"] = type(exc).__name__
            (raw / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise
        finally:
            record["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_json(directory / "result.json", record)
    finally:
        lock.unlink(missing_ok=True)


def main(run):
    parser = argparse.ArgumentParser(description="Local AIMO experiment. No automatic downloads or GPU jobs.")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Group-split a LOCAL JSON/JSONL/Parquet source")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--validation-fraction", type=float, default=.2)
    p = commands.add_parser("run", help="Extract, train, evaluate, record")
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default="main")
    p.add_argument("--name", required=True)
    p.add_argument("--layers", type=int, nargs="+", default=[16])
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--template", choices=["plain", "chat"], default="chat")
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--width", type=int, default=0)
    p.add_argument("--learning-rate", type=float, default=.001)
    p.add_argument("--threshold", type=float, default=.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-seconds", type=float, default=300)
    p = commands.add_parser("collect", help="Generate answers and archive bounded activation observations")
    p.add_argument("--originals", type=Path, default=ROOT / "data" / "collection_original_tasks.json")
    p.add_argument("--variations", type=Path, default=ROOT / "variations_tasks.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    p.add_argument("--revision", default="6e8885a6ff5c1dc5201574c8fd700323f23c25fa")
    p.add_argument("--device", default="cuda", choices=["cuda"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16"])
    p.add_argument("--max-prompt-tokens", type=int, default=8192)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--selection-seed", type=int, default=20260910)
    p.add_argument("--do-sample", action="store_true", help="sample instead of the default greedy decode")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--allow-download", action="store_true", help="allow first-run model/tokenizer download")
    p.add_argument("--cpu-offload", action="store_true", help="stage weights through CPU/disk via Accelerate")
    p.add_argument("--gpu-memory-gib", type=int)
    p.add_argument("--offload-folder", type=Path, default=ROOT / ".runtime" / "offload")
    p = commands.add_parser("export", help="Package exact run sources plus head, not the whole repo")
    p.add_argument("--name", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "export":
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.name):
            parser.error("Invalid run name")
        export(args)
    elif args.command == "collect":
        if min(args.max_prompt_tokens, args.max_new_tokens) < 1 or args.temperature <= 0:
            parser.error("prompt/generation limits and temperature must be positive")
        if not args.originals.is_file() or not args.variations.is_file():
            parser.error("originals and variations files must exist")
        from research.experiment import collect
        collect(args)
    else:
        import math
        if min(args.epochs, args.batch_size, args.max_length) < 1 or args.width < 0:
            parser.error("Positive lengths/batches/epochs and nonnegative width required")
        if not (math.isfinite(args.learning_rate) and args.learning_rate > 0 and
                math.isfinite(args.max_seconds) and args.max_seconds > 0 and 0 <= args.threshold <= 1):
            parser.error("Invalid learning rate, threshold or time limit")
        with journal(args) as record:
            run(args, record)
        print(canonical({k: record[k] for k in ("name", "status", "accuracy", "majority_accuracy")}))
