#!/usr/bin/env python3
"""Validate the machine-readable contract for externally generated paraphrases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class ResponseValidationError(RuntimeError):
    pass


def request_families(payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict) or set(payload) != {"tasks"} or not isinstance(payload["tasks"], list) or not payload["tasks"]:
        raise ResponseValidationError("request top level must be exactly a non-empty {'tasks': [...]} object")
    expected = {}
    for number, task in enumerate(payload["tasks"], start=1):
        if not isinstance(task, dict) or set(task) != {"family_id", "original_problem"}:
            raise ResponseValidationError(f"request task {number} has an invalid schema")
        family_id, original = task["family_id"], task["original_problem"]
        if not isinstance(family_id, str) or not family_id or family_id in expected:
            raise ResponseValidationError(f"request task {number} has a missing or duplicate family_id")
        if not isinstance(original, str) or not original.strip():
            raise ResponseValidationError(f"request task {number} has an empty original_problem")
        expected[family_id] = original
    return expected


def validate(payload: Any, expected: dict[str, str]) -> int:
    if not isinstance(payload, dict) or set(payload) != {"tasks"} or not isinstance(payload["tasks"], list) or not payload["tasks"]:
        raise ResponseValidationError("response top level must be exactly a non-empty {'tasks': [...]} object")
    families = set()
    variants = 0
    for task_number, task in enumerate(payload["tasks"], start=1):
        if not isinstance(task, dict) or set(task) != {"family_id", "variants"}:
            raise ResponseValidationError(f"task {task_number} must contain only family_id and variants")
        family_id = task["family_id"]
        if not isinstance(family_id, str) or not family_id or family_id in families:
            raise ResponseValidationError(f"task {task_number} has a missing or duplicate family_id")
        if family_id not in expected:
            raise ResponseValidationError(f"task {task_number} has an unexpected family_id")
        families.add(family_id)
        rows = task["variants"]
        if not isinstance(rows, list) or len(rows) != 10:
            raise ResponseValidationError(f"task {task_number} must have exactly 10 variants")
        expected_ids = {f"v{number:02d}" for number in range(1, 11)}
        seen_ids = set()
        seen_problems = set()
        for row_number, row in enumerate(rows, start=1):
            if not isinstance(row, dict) or set(row) != {"variant_id", "problem"}:
                raise ResponseValidationError(f"task {task_number}, variant {row_number} has an invalid schema")
            variant_id, problem = row["variant_id"], row["problem"]
            if variant_id not in expected_ids or variant_id in seen_ids:
                raise ResponseValidationError(f"task {task_number} has invalid or duplicate variant_id")
            normalized_problem = problem.strip() if isinstance(problem, str) else ""
            if not normalized_problem or normalized_problem in seen_problems:
                raise ResponseValidationError(f"task {task_number} has an empty or duplicate problem text")
            if normalized_problem == expected[family_id].strip():
                raise ResponseValidationError(f"task {task_number} repeats its original_problem")
            seen_ids.add(variant_id)
            seen_problems.add(normalized_problem)
            variants += 1
        if seen_ids != expected_ids:
            raise ResponseValidationError(f"task {task_number} must use v01 through v10")
    if families != set(expected):
        missing = len(set(expected) - families)
        unexpected = len(families - set(expected))
        raise ResponseValidationError(f"response family_id set does not match request (missing={missing}, unexpected={unexpected})")
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated paraphrase JSON.")
    parser.add_argument("request_json", type=Path)
    parser.add_argument("response_json", type=Path)
    args = parser.parse_args()
    try:
        request = json.loads(args.request_json.read_text(encoding="utf-8"))
        payload = json.loads(args.response_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResponseValidationError(f"could not read JSON: {exc}") from exc
    variants = validate(payload, request_families(request))
    print(f"valid: {len(payload['tasks'])} families, {variants} variants")


if __name__ == "__main__":
    main()
