"""Check guide links, Python syntax, and verbatim function snapshots (stdlib only)."""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

GUIDE = Path(__file__).resolve().parents[1]
ROOT = GUIDE.parent


def find_symbol(tree: ast.AST, qualified_name: str) -> ast.AST:
    node = tree
    for part in qualified_name.split("."):
        matches = [child for child in getattr(node, "body", [])
                   if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                   and child.name == part]
        if len(matches) != 1:
            raise ValueError(f"Missing or ambiguous symbol: {qualified_name}")
        node = matches[0]
    return node


def check_snapshots(guide: Path = GUIDE) -> list[str]:
    errors = []
    manifest = json.loads((guide / "annotated/source_map.json").read_text(encoding="utf-8"))
    for snapshot in manifest["snapshots"]:
        try:
            source = ast.parse((guide.parent / snapshot["source"]).read_text(encoding="utf-8"))
            copy = ast.parse((guide / snapshot["file"]).read_text(encoding="utf-8"))
            for copied_name, source_name in snapshot["symbols"].items():
                left = ast.dump(find_symbol(copy, copied_name), include_attributes=False)
                right = ast.dump(find_symbol(source, source_name), include_attributes=False)
                if left != right:
                    errors.append(f"Changed source: {snapshot['source']} :: {source_name}")
        except (OSError, SyntaxError, ValueError) as exc:
            errors.append(f"Snapshot {snapshot['file']}: {exc}")
    return errors


def check_links(guide: Path = GUIDE, known_paths: set[Path] | None = None) -> list[str]:
    """Check file targets; optional known_paths supports a fetched Git tree inventory.

    Deliberately not a complete Markdown parser: these documents use inline links.
    External URLs and heading fragments are not checked over the network.
    """
    errors = []
    for document in guide.rglob("*.md"):
        text = re.sub(r"```.*?```", "", document.read_text(encoding="utf-8"), flags=re.S)
        for link in re.findall(r"\[[^\]\n]*\]\(([^)\s]+)\)", text):
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            target = (document.parent / unquote(parsed.path)).resolve()
            if not target.is_relative_to(guide.parent.resolve()):
                errors.append(f"Link outside repository: {document.name}: {link}")
            elif not target.exists() and (known_paths is None or target not in known_paths):
                errors.append(f"Missing link: {document.relative_to(guide)}: {link}")
    return errors


def check_syntax(guide: Path = GUIDE) -> list[str]:
    errors = []
    for path in guide.rglob("*.py"):
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as exc:
            errors.append(f"Syntax: {path.relative_to(guide)}: {exc}")
    return errors


def main() -> int:
    errors = check_links() + check_snapshots() + check_syntax()
    for message in errors:
        print(message, file=sys.stderr)
    if errors:
        print(f"FAILED: {len(errors)} issue(s)", file=sys.stderr)
        return 1
    words = sum(len(path.read_text(encoding="utf-8").split()) for path in (GUIDE / "route").glob("*.md"))
    print(f"OK: local file links, Python syntax, 6 function snapshots. Main route: {words} words.")
    print("This does not run the LLM or validate current competition rules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
