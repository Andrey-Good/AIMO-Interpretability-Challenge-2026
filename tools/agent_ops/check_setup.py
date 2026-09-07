"""Static project self-check; no network, model calls, or heavy dependencies."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def check(root: Path = ROOT) -> list[str]:
    errors = []
    scopes = [root / p for p in ("research", "tools/agent_ops", "docs/agents", "MEMORY", ".agents", ".codex")]
    files = [p for d in scopes for p in d.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    files.append(root / "AGENTS.md")
    for path in files:
        try:
            if path.suffix == ".py":
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            elif path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
            elif path.suffix == ".toml":
                tomllib.loads(path.read_text(encoding="utf-8"))
            elif path.suffix == ".md":
                text = path.read_text(encoding="utf-8")
                # Inline Markdown links to local files; not external URLs/header anchors.
                for target in re.findall(r"\]\(([^)\s]+)\)", text):
                    if "://" in target or target.startswith("#"):
                        continue
                    linked = (path.parent / target.split("#")[0]).resolve()
                    if not linked.is_relative_to(root.resolve()) or not linked.exists():
                        errors.append(f"broken/escaping link: {path.relative_to(root)} -> {target}")
        except (SyntaxError, ValueError, OSError) as exc:
            errors.append(f"{path}: {exc}")
    config = tomllib.loads((root / ".codex/config.toml").read_text())
    if "profiles" in config or "profile" in config:
        errors.append("project-local profiles are ignored by current Codex")
    if config.get("sandbox_mode") != "workspace-write" or config.get("approval_policy") != "on-request":
        errors.append("unexpected default permission policy")
    if config.get("agents", {}).get("max_concurrent_threads_per_session") != 2:
        errors.append("unexpected concurrency policy; review test and documented budget")
    for path in (root / ".codex/agents").glob("*.toml"):
        role = tomllib.loads(path.read_text())
        for key in ("name", "description", "developer_instructions", "model", "model_reasoning_effort"):
            if not role.get(key):
                errors.append(f"{path.name}: missing {key}")
        if role.get("model_reasoning_effort") not in {"low", "medium", "high", "xhigh", "max"}:
            errors.append(f"{path.name}: invalid reasoning effort")
    names = set()
    for path in (root / ".agents/skills").glob("*/SKILL.md"):
        text = path.read_text()
        header = text.split("---", 2)
        if len(header) != 3 or not re.search(r"^name: .+", header[1], re.M) or not re.search(r"^description: .+", header[1], re.M):
            errors.append(f"invalid skill metadata: {path}")
        names.add(path.parent.name)
    if len(names) != 11:
        errors.append("expected 10 AIMO skills and 1 official define-goal skill")
    examples = json.loads((root / ".agents/skills/trigger_cases.json").read_text())
    if {e['skill'] for e in examples} != names - {"define-goal"} or not all(e.get("positive") and e.get("negative") for e in examples):
        errors.append("incomplete trigger examples")
    path = root / ".agents/skills/define-goal/SKILL.md"
    content = path.read_bytes()
    digest = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    if digest != "87f111bd700e0d993465f7ac741847b5daee57d6":
        errors.append("official skill differs from pinned upstream; update provenance explicitly")
    if (root / "AGENTS.md").stat().st_size > 16384:
        errors.append("root AGENTS too large; move detail into skills/docs")
    return errors


if __name__ == "__main__":
    errors = check()
    print("\n".join(errors) if errors else "PASS: syntax, local links, agent/skill metadata, pinned skill, config policy")
    sys.exit(bool(errors))
