"""Run ONE reviewed experiment; journal outcomes without staging, committing or pushing.

This is a cooperative runner, NOT a security boundary, VRAM limiter or API budget.
The lock covers worktrees sharing one Git common-dir, not other clones/hosts.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any


class PlanError(ValueError):
    """Reject an invalid/unapproved plan before starting any experiment."""


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, obj: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def within(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise PlanError("path escapes repository")
    return path


def validate_plan(plan: dict[str, Any], root: Path) -> None:
    if plan.get("schema_version") != 1:
        raise PlanError("unsupported plan version")
    if not isinstance(plan.get("run_id"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", plan["run_id"]):
        raise PlanError("invalid run_id")
    approval = plan.get("approval", {})
    if not isinstance(approval, dict) or approval.get("status") != "approved" or not approval.get("approved_by"):
        raise PlanError("explicit human approval must be recorded")
    for field in ("hypothesis", "dataset_revision", "split_id", "feature_spec", "predictor_spec", "primary_metric"):
        value = plan.get(field)
        if not isinstance(value, str) or not value.strip() or "UNSET" in value:
            raise PlanError(f"missing concrete {field}")
    timeout = plan.get("timeout_seconds")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 86400:
        raise PlanError("timeout_seconds must be >0 and <=86400")
    argv = plan.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x and "\0" not in x for x in argv):
        raise PlanError("argv must be a nonempty string list")
    resource = plan.get("resource", {})
    if not isinstance(resource, dict) or resource.get("kind") not in ("cpu", "gpu"):
        raise PlanError("resource.kind must be cpu or gpu")
    if resource["kind"] == "gpu":
        devices = resource.get("cuda_visible_devices")
        if not isinstance(devices, str) or not re.fullmatch(r"\d+(,\d+)*", devices):
            raise PlanError("GPU device IDs must be explicitly approved")
    if type(plan.get("seed")) is not int:
        raise PlanError("seed must be an integer; experiment code must actually use it")
    outputs = plan.get("expected_outputs", [])
    if not isinstance(outputs, list) or not all(isinstance(p, str) and p for p in outputs):
        raise PlanError("expected_outputs must contain relative file paths")
    for p in outputs:
        if Path(p).is_absolute():
            raise PlanError("expected output must be relative")
        within(root, p)


def terminate_tree(process: subprocess.Popen) -> None:
    """Kill the owned process group on POSIX; Windows needs host verification."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            time.sleep(0.05)
            # Also kill descendants if the leader already exited.
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def execute(plan_path: Path, root: Path | None = None) -> dict[str, Any]:
    root = (root or Path(git(Path.cwd(), "rev-parse", "--show-toplevel"))).resolve()
    plan_path = plan_path.resolve()
    if not plan_path.is_relative_to(root):
        raise PlanError("plan must be inside repository")
    raw = plan_path.read_bytes()
    plan = json.loads(raw)
    if not isinstance(plan, dict):
        raise PlanError("plan must be a JSON object")
    validate_plan(plan, root)
    if git(root, "status", "--porcelain", "--untracked-files=all"):
        raise PlanError("working tree is dirty: commit reviewed code/plan or previous result first")
    relplan = plan_path.relative_to(root).as_posix()
    git(root, "ls-files", "--error-unmatch", "--", relplan)
    if any(within(root, p).exists() for p in plan.get("expected_outputs", [])):
        raise PlanError("expected output already exists; choose a new output path")
    code_sha = git(root, "rev-parse", "HEAD")
    common = Path(git(root, "rev-parse", "--git-common-dir"))
    if not common.is_absolute():
        common = root / common
    lock = common.resolve() / "aimo-experiment.lock"
    record_dir = root / "MEMORY" / "runs" / plan["run_id"]
    log_dir = root / ".runtime" / plan["run_id"]
    if record_dir.exists() or log_dir.exists():
        raise PlanError("run_id already exists; never overwrite a previous attempt")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise PlanError("another run or stale lock exists; inspect, do not delete blindly") from exc
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump({"pid": os.getpid(), "run_id": plan["run_id"], "started": utc_now()}, stream)
        record_dir.mkdir(parents=True, exist_ok=False)
        log_dir.mkdir(parents=True, exist_ok=False)
        record_path = record_dir / "execution.json"
        argv = [sys.executable if a == "{python}" else a for a in plan["argv"]]
        record: dict[str, Any] = {
            "schema_version": 1, "run_id": plan["run_id"], "status": "running",
            "scientific_status": "unreviewed", "code_sha": code_sha,
            "plan_path": relplan, "plan_sha256": hashlib.sha256(raw).hexdigest(),
            "argv": argv, "started_at": utc_now(), "python": sys.version,
            "platform": sys.platform, "resource": plan["resource"],
            "stdout": str((log_dir / "stdout.log").relative_to(root)),
            "stderr": str((log_dir / "stderr.log").relative_to(root)),
            "exit_code": None,
        }
        atomic_json(record_path, record)
        started = time.monotonic()
        process = None
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "" if plan["resource"]["kind"] == "cpu" else plan["resource"]["cuda_visible_devices"]
        try:
            with (log_dir / "stdout.log").open("wb") as out, (log_dir / "stderr.log").open("wb") as err:
                process = subprocess.Popen(argv, cwd=root, env=env, stdout=out, stderr=err,
                                           start_new_session=os.name != "nt")
                process.wait(timeout=plan["timeout_seconds"])
                record["status"] = "succeeded" if process.returncode == 0 else "failed"
                record["exit_code"] = process.returncode
        except subprocess.TimeoutExpired:
            terminate_tree(process)
            record["status"] = "timed_out"
            record["exit_code"] = process.returncode
        except KeyboardInterrupt:
            if process is not None:
                terminate_tree(process)
            record["status"] = "interrupted"
            record["exit_code"] = None if process is None else process.returncode
        except OSError as exc:
            record["status"] = "failed"
            record["error_type"] = type(exc).__name__
            record["error"] = str(exc)
        finally:
            record["finished_at"] = utc_now()
            record["elapsed_seconds"] = time.monotonic() - started
            record["head_after"] = git(root, "rev-parse", "HEAD")
            record["tracked_changes_after"] = git(root, "diff", "--name-only", "HEAD").splitlines()
            record["missing_outputs"] = [p for p in plan.get("expected_outputs", []) if not within(root, p).is_file()]
            if record["status"] == "succeeded" and record["missing_outputs"]:
                record["status"] = "missing_outputs"
            if record["status"] == "succeeded" and (record["head_after"] != code_sha or record["tracked_changes_after"]):
                record["status"] = "source_changed"
            record["outputs"] = []
            for name in plan.get("expected_outputs", []):
                path = within(root, name)
                if path.is_file():
                    digest = hashlib.sha256()
                    with path.open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                    record["outputs"].append({"path": name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
            atomic_json(record_path, record)
        return record
    finally:
        lock.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    try:
        record = execute(args.plan)
    except (PlanError, OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Not started: {exc}") from exc
    print(json.dumps(record, ensure_ascii=False, indent=2))
    raise SystemExit(0 if record["status"] == "succeeded" else 1)


if __name__ == "__main__":
    main()
