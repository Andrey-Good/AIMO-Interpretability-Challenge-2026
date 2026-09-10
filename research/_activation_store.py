"""Bounded, resumable storage primitives for activation collection.

This module contains no model loading or scientific classification.  A collector
writes one input per ``.partial.h5`` file and atomically renames it only after
its manifest validates; the index is rebuilt from completed files.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import torch


def require_cuda() -> None:
    """Fail before tokenizer/model loading when a CPU PyTorch wheel is installed."""
    if torch.version.cuda is None or not torch.cuda.is_available():
        raise RuntimeError(
            "collect requires an NVIDIA CUDA PyTorch wheel and visible GPU; run `uv sync` "
            "with the project's pytorch-cu130 index on the target machine, then check nvidia-smi"
        )


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def bf16_bits(value: torch.Tensor) -> torch.Tensor:
    """Return native BF16 bits; conversion through FP32 is deliberately forbidden."""
    if value.dtype != torch.bfloat16:
        raise ValueError("raw activation must have native bfloat16 dtype")
    return value.detach().cpu().contiguous().view(torch.uint16)


def bits_bf16(bits: torch.Tensor) -> torch.Tensor:
    if bits.dtype != torch.uint16:
        raise ValueError("raw activation bits must be uint16")
    return bits.contiguous().view(torch.bfloat16)


GEOMETRY_CHANNELS = ("r2", "a2", "m2", "ra", "rm", "am", "a_plus_m2", "h2", "u2", "ru", "uh", "rh", "h_minus_r2")


def geometry_channels(r: torch.Tensor, a: torch.Tensor, m: torch.Tensor, u: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Return the contract's 13 FP32 scalars per token, preserving actual rounded u/h."""
    tensors = [x.detach().float() for x in (r, a, m, u, h)]
    if any(x.shape != tensors[0].shape or x.ndim < 1 for x in tensors) or not all(torch.isfinite(x).all() for x in tensors):
        raise ValueError("finite equal-shaped hook tensors required")
    r, a, m, u, h = tensors
    dot = lambda x, y: (x * y).sum(-1)
    return torch.stack((dot(r,r),dot(a,a),dot(m,m),dot(r,a),dot(r,m),dot(a,m),dot(a+m,a+m),
                        dot(h,h),dot(u,u),dot(r,u),dot(u,h),dot(r,h),dot(h-r,h-r)), -1)


@dataclass
class Welford:
    """Per-coordinate FP32 moments, with undefined empty phases represented by n=0."""
    n: int
    mean: torch.Tensor
    m2: torch.Tensor

    @classmethod
    def empty(cls, width: int) -> "Welford":
        return cls(0, torch.zeros(width, dtype=torch.float32), torch.zeros(width, dtype=torch.float32))

    def update(self, rows: torch.Tensor) -> None:
        if rows.ndim != 2 or rows.shape[1] != self.mean.numel():
            raise ValueError("expected [tokens, hidden_width]")
        if not torch.isfinite(rows).all():
            raise ValueError("nonfinite activation")
        # A collector runs forwards in inference_mode; clone converts its special
        # immutable tensor into normal CPU storage before Welford's in-place math.
        rows = rows.detach().to(dtype=torch.float32, device="cpu").clone()
        for row in rows:
            self.n += 1
            delta = row - self.mean
            self.mean += delta / self.n
            self.m2 += delta * (row - self.mean)

    def merge(self, other: "Welford") -> "Welford":
        if self.mean.shape != other.mean.shape:
            raise ValueError("moment widths differ")
        if not other.n:
            return self
        if not self.n:
            self.n, self.mean, self.m2 = other.n, other.mean.clone(), other.m2.clone()
            return self
        n = self.n + other.n
        delta = other.mean - self.mean
        self.m2 += other.m2 + delta.square() * (self.n * other.n / n)
        self.mean += delta * (other.n / n)
        self.n = n
        return self


def plan_cases(original: dict, variations: dict) -> list[dict]:
    """Create fixed 137*11 case identities; labels are intentionally absent."""
    def tasks(payload: dict, fields: set[str]) -> list[dict]:
        if set(payload) != {"tasks"} or not isinstance(payload["tasks"], list):
            raise ValueError("invalid task payload")
        if not payload["tasks"] or any(set(t) != fields for t in payload["tasks"]):
            raise ValueError("invalid task schema")
        return payload["tasks"]
    originals = tasks(original, {"family_id", "original_problem"})
    variants = tasks(variations, {"family_id", "variants"})
    by_id = {t["family_id"]: t for t in variants}
    if len(by_id) != len(variants) or set(by_id) != {t["family_id"] for t in originals}:
        raise ValueError("original/variation family IDs differ")
    cases = []
    for task in originals:
        family = task["family_id"]
        cases.append({"case_id": f"{family}:original", "family_id": family, "variant_id": "original", "text": task["original_problem"]})
        rows = by_id[family]["variants"]
        if len(rows) != 10 or {r.get("variant_id") for r in rows} != {f"v{i:02d}" for i in range(1, 11)}:
            raise ValueError(f"{family}: require exactly v01..v10")
        for row in sorted(rows, key=lambda r: r["variant_id"]):
            text = row.get("problem")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{family}: empty variation")
            cases.append({"case_id": f"{family}:{row['variant_id']}", "family_id": family, "variant_id": row["variant_id"], "text": text})
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("duplicate case ID")
    return cases


def write_completed(path: Path, manifest: dict, datasets: dict[str, torch.Tensor]) -> Path:
    """Write a closed HDF5 file or fail; callers never index partial files."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("activation collection needs h5py; run uv sync") from exc
    if path.exists() or path.with_suffix(".partial.h5").exists():
        raise FileExistsError("refusing to overwrite an activation run")
    partial = path.with_suffix(".partial.h5")
    payload = {}
    for name, tensor in datasets.items():
        if tensor.device.type != "cpu":
            raise ValueError("storage receives CPU tensors only")
        array = tensor.contiguous().numpy()
        payload[name] = {"shape": list(array.shape), "dtype": str(array.dtype),
                         "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    partial.parent.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(partial, "w") as h5:
            h5.attrs["manifest"] = canonical(manifest)
            h5.attrs["manifest_sha256"] = sha256(manifest)
            h5.attrs["payload"] = canonical(payload)
            for name, tensor in datasets.items():
                h5.create_dataset(name, data=tensor.numpy(), chunks=True)
        with h5py.File(partial, "r") as h5:
            if h5.attrs["manifest_sha256"] != sha256(manifest) or set(h5.keys()) != set(datasets):
                raise ValueError("incomplete activation file")
            _verify_payload(h5, payload)
        partial.replace(path)
    except BaseException:
        # Keep a partial for forensic inspection/retry; it is never considered complete.
        raise
    return path


def completed_manifest(path: Path) -> dict:
    """Read one closed archive only after verifying every payload dataset."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("activation collection needs h5py; run uv sync") from exc
    if path.name.endswith(".partial.h5"):
        raise ValueError("partial archives are never complete")
    with h5py.File(path, "r") as h5:
        manifest = json.loads(h5.attrs["manifest"])
        if h5.attrs["manifest_sha256"] != sha256(manifest):
            raise ValueError(f"corrupt manifest: {path}")
        try:
            payload = json.loads(h5.attrs["payload"])
        except (KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"missing payload manifest: {path}") from exc
        _verify_payload(h5, payload)
        return manifest


def completed_manifests(directory: Path) -> Iterable[dict]:
    """Yield only valid closed files; corrupted/mismatched runs cannot be skipped."""
    for path in sorted(directory.glob("*.h5")):
        if not path.name.endswith(".partial.h5"):
            yield completed_manifest(path)


def _verify_payload(h5, payload: dict) -> None:
    if set(h5.keys()) != set(payload):
        raise ValueError("dataset set differs from payload manifest")
    for name, expected in payload.items():
        dataset = h5[name]
        if list(dataset.shape) != expected["shape"] or str(dataset.dtype) != expected["dtype"]:
            raise ValueError(f"dataset schema mismatch: {name}")
        if hashlib.sha256(dataset[...].tobytes()).hexdigest() != expected["sha256"]:
            raise ValueError(f"dataset checksum mismatch: {name}")
