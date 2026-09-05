"""Явно перевести старые локальные cases в строковый формат ingestion.

Сохраняет ID и порядок; не читает labels и не изменяет исходник.
Новый файл никогда не перезаписывает существующий. Только стандартная библиотека.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def normalize_lines(lines: Iterable[str]) -> tuple[list[dict], int]:
    rows: list[dict] = []
    seen: set[str] = set()
    changed = 0
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Строка {number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Строка {number}: ожидается JSON object")
        for key in ("id", "model_id"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"Строка {number}: неверный {key}")
        if row["id"] in seen:
            raise ValueError(f"Строка {number}: повторный id {row['id']!r}")
        seen.add(row["id"])
        problem = row.get("problem")
        if isinstance(problem, dict):
            problem = problem.get("original_problem")
            changed += 1
        if not isinstance(problem, str) or not problem:
            raise ValueError(f"Строка {number}: нужен непустой problem/original_problem")
        rows.append({**row, "problem": problem})
    if not rows:
        raise ValueError("Входной файл пуст")
    return rows, changed


def normalize_file(source: Path, destination: Path) -> tuple[int, int]:
    if source.resolve() == destination.resolve():
        raise ValueError("Исходник и выход должны быть разными файлами")
    with source.open(encoding="utf-8") as stream:
        rows, changed = normalize_lines(stream)
    # Валидация завершается ДО создания выхода. 'x' также защищает от гонки.
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
    return len(rows), changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        count, changed = normalize_file(args.source, args.destination)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Не удалось преобразовать cases: {exc}\n")
    print(f"Записано {count} строк; преобразовано {changed}; выход: {args.destination}")


if __name__ == "__main__":
    main()
