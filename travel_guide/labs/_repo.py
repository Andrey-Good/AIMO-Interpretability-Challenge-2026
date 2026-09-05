"""Helpers for reading trusted repository code without loading an LLM.

load_functions executes only the named source functions, not module-level imports.
This is a test/teaching helper, NOT the way a submission loads its implementation.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def load_module(relative_path: str, name: str) -> ModuleType:
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load repository module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def load_functions(relative_path: str, names: list[str], namespace: dict[str, Any]) -> dict[str, Any]:
    """Extract trusted top-level functions so tests need no Transformers import."""
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    missing = set(names) - found.keys()
    if missing:
        raise RuntimeError(f"Source functions moved or disappeared: {sorted(missing)}")
    future = ast.parse("from __future__ import annotations").body[0]
    selected = ast.Module(body=[future, *(found[name] for name in names)], type_ignores=[])
    scope = dict(namespace)
    exec(compile(ast.fix_missing_locations(selected), str(path), "exec"), scope)
    return scope


def read_literal(relative_path: str, name: str) -> Any:
    """Read a literal constant without executing imports or deserializing files."""
    path = ROOT / relative_path
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"Literal {name} not found in {path}")
