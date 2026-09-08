#!/usr/bin/env python3
"""Export original AIMO problems into label-free paraphrase request batches.

The source dataset remains the authority for rows and old robustness labels.
Only a family ID and its original statement are exported, so the request files
can safely be given to a text-generation system without revealing labels,
answers, models, or existing perturbations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from import_hf_dataset import fetch_all_rows, load_source_json, request_headers


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "aimo-interp/augmented-sample-math-agg"
DEFAULT_REVISION = "f972ced0705096f8d7ca7fac30825900b8b7fb6a"
DEFAULT_CONFIG = "default"
DEFAULT_SPLIT = "validation"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "paraphrase-preparation"
VARIANTS_PER_FAMILY = 10
PILOT_FAMILIES = 5
STUDY_FAMILIES = 50
FAMILIES_PER_FILE = 5

GENERATOR_PROMPT = """Мы исследуем, как меняются ответы и внутренние активации языковой модели, когда одна и та же математическая задача сформулирована по-разному. Цель — выбрать признаки для классификатора устойчивости.

Для каждой записи входного JSON создай ровно 10 переформулировок. Сохрани язык, математическое содержание, числа, все ограничения и искомую величину исходной задачи. Не добавляй ответы, решения, подсказки, метки, сведения о моделях или комментарии.

Верни только полный машиночитаемый JSON строго такой формы:
{"tasks":[{"family_id":"f000001","variants":[{"variant_id":"v01","problem":"перефраз"}]}]}
В каждом `variants` верни ровно 10 объектов с номерами `v01`…`v10` и сохрани исходный `family_id`.
"""


class PreparationError(RuntimeError):
    """Raised when source rows cannot form a reproducible request package."""


def required_string(row: dict[str, Any], name: str, row_number: int) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value:
        raise PreparationError(f"source row {row_number} has invalid {name}")
    return value


def family_id(dataset_id: str, problem_id: str) -> str:
    """Return a stable opaque identifier for a dataset/problem family."""

    identity = json.dumps([dataset_id, problem_id], ensure_ascii=False, separators=(",", ":"))
    return "f" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]


def source_digest(rows: list[dict[str, Any]]) -> str:
    canonical = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def payload_digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def fetch_main_revision_sha(dataset: str) -> str:
    request = urllib.request.Request(
        f"https://huggingface.co/api/datasets/{dataset}/revision/main", headers=request_headers()
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        raise PreparationError(f"could not verify dataset main revision: {exc}") from exc
    sha = payload.get("sha") if isinstance(payload, dict) else None
    if not isinstance(sha, str) or not sha:
        raise PreparationError("dataset main revision response has no sha")
    return sha


def normalise_families(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Group rows by the existing ``dataset_id`` + ``problem_id`` family key.

    A conflicting text or historic label is a source-data error, not a reason
    to silently select a different subset.
    """

    models = set()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for number, row in enumerate(rows, start=1):
        dataset_id = required_string(row, "dataset_id", number)
        problem_id = required_string(row, "problem_id", number)
        original_problem = required_string(row, "original_problem", number)
        model_id = required_string(row, "model_id", number)
        models.add(model_id)
        label = row.get("model_is_robust")
        if type(label) is not bool:
            raise PreparationError(f"source row {number} has invalid model_is_robust")
        grouped[(dataset_id, problem_id)].append(
            {"original_problem": original_problem, "model_is_robust": label, "model_id": model_id}
        )

    if len(models) != 1:
        raise PreparationError(f"source contains {len(models)} model_id values; refusing to mix historic labels")

    families = []
    conflicting_label_families = 0
    conflicting_text_families = 0
    for (dataset_id, problem_id), members in sorted(grouped.items()):
        texts = {member["original_problem"] for member in members}
        labels = {member["model_is_robust"] for member in members}
        if len(texts) != 1:
            conflicting_text_families += 1
            # There is no unambiguous text to send out; fail rather than choose one.
            raise PreparationError(
                f"family {(dataset_id, problem_id)!r} has conflicting original_problem texts"
            )
        if len(labels) != 1:
            conflicting_label_families += 1
            raise PreparationError(f"family {(dataset_id, problem_id)!r} has conflicting historic labels")
        historic_label = labels.pop()
        families.append(
            {
                "family_id": family_id(dataset_id, problem_id),
                "dataset_id": dataset_id,
                "problem_id": problem_id,
                "original_problem": texts.pop(),
                "historic_label": historic_label,
                "source_rows": len(members),
                "model_id": members[0]["model_id"],
            }
        )
    return families, {
        "source_rows": len(rows),
        "families": len(families),
        "conflicting_label_families": conflicting_label_families,
        "conflicting_text_families": conflicting_text_families,
    }


def deterministic_order(families: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(families, key=lambda family: hashlib.sha256(family["family_id"].encode()).hexdigest())


def select_study_families(families: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_label: dict[bool, list[dict[str, Any]]] = {False: [], True: []}
    for family in families:
        label = family["historic_label"]
        if label is not None:
            by_label[label].append(family)
    per_label = STUDY_FAMILIES // 2
    false_families = deterministic_order(by_label[False])
    true_families = deterministic_order(by_label[True])
    selected = false_families[:per_label] + true_families[:per_label]
    if len(selected) != STUDY_FAMILIES:
        counts = {str(label).lower(): len(items) for label, items in by_label.items()}
        raise PreparationError(
            f"need {per_label} unambiguous families per historic label for the study; have {counts}"
        )
    # The pilot has the requested 3/2 historic-label mix.  The rest of the
    # study follows it in a deterministic, label-mixed order.
    pilot = true_families[:3] + false_families[:2]
    remaining = true_families[3:per_label] + false_families[2:per_label]
    return deterministic_order(pilot) + deterministic_order(remaining)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def request_payload(families: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "tasks": [
            {"family_id": family["family_id"], "original_problem": family["original_problem"]}
            for family in families
        ]
    }


def assert_label_free(payload: dict[str, Any]) -> None:
    forbidden = {"model_is_robust", "historic_label", "model_id", "answer", "solution", "permutation_type"}
    def field_names(value: Any) -> set[str]:
        if isinstance(value, dict):
            result = set(value)
            for item in value.values():
                result.update(field_names(item))
            return result
        if isinstance(value, list):
            result = set()
            for item in value:
                result.update(field_names(item))
            return result
        return set()
    if forbidden.intersection(field_names(payload)):
        raise PreparationError("request payload contains a forbidden source-only field")


def write_index(output_dir: Path, *, all_count: int, study_count: int) -> None:
    content = f"""# Набор для перефразирования

`source/all_original_tasks.json` содержит {all_count} исходных задач.
`requests/study50.json` — полный стартовый набор из {study_count} семейств, а
`requests/batches/` разбивает его на запросы по {FAMILIES_PER_FILE} задач;
`batch-01.json` — пилот из {PILOT_FAMILIES} семейств и уже входит в study50.
В каждом передаваемом генератору JSON только `family_id` и `original_problem`:
меток, ответов, моделей и старых возмущений нет.

Исторические метки относятся к `qwen3-8b:low`: study50 намеренно содержит
25+25 семейств и 500 новых текстов, а пилот — 3+2 и 50. Это не естественное
распределение и не доказанный достаточный размер для классификатора. При
изменении `prompt.md` пилот нужно получить заново.

Передавать генератору следует один request JSON вместе с `prompt.md`. Его ответ
нужно сохранить отдельно и проверить из worktree командой
`python C:\\Users\\urako\\.codex\\worktrees\\2d48\\AIMO-Interpretability-Challenge-2026\\scripts\\validate_paraphrase_response.py <request.json> <ответ.json>`.
Валидатор проверяет структуру и идентификаторы, но не смысловую эквивалентность.
"""
    atomic_write(output_dir / "README.md", content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare label-free paraphrase request JSON files.")
    parser.add_argument("--source-json", type=Path, help="saved Hugging Face rows JSON; avoids network access")
    parser.add_argument("--fetch", action="store_true", help="fetch the pinned public dataset")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.source_json) == args.fetch:
        raise PreparationError("provide exactly one of --source-json or --fetch")
    if args.source_json:
        rows = load_source_json(args.source_json)
        verification_digest = None
        main_sha_before = None
        main_sha_after = None
    else:
        main_sha_before = fetch_main_revision_sha(args.dataset)
        if main_sha_before != args.revision:
            raise PreparationError(f"dataset main revision {main_sha_before} is not requested pin {args.revision}")
        rows = fetch_all_rows(args.dataset, args.revision, args.config, args.split)
        first_digest = source_digest(rows)
        # Re-read the pinned rows endpoint after pagination and refuse an
        # export if it did not yield byte-for-byte equivalent source records.
        verified_rows = fetch_all_rows(args.dataset, args.revision, args.config, args.split)
        verification_digest = source_digest(verified_rows)
        if verification_digest != first_digest:
            raise PreparationError("pinned source rows changed between verification reads")
        main_sha_after = fetch_main_revision_sha(args.dataset)
        if main_sha_after != args.revision:
            raise PreparationError(f"dataset main revision changed to {main_sha_after} during paging")
    families, summary = normalise_families(rows)
    selected = select_study_families(families)
    all_payload = request_payload(deterministic_order(families))
    assert_label_free(all_payload)
    write_json(args.output_dir / "source" / "all_original_tasks.json", all_payload)
    study_payload = request_payload(selected)
    assert_label_free(study_payload)
    write_json(args.output_dir / "requests" / "study50.json", study_payload)
    for start in range(0, len(selected), FAMILIES_PER_FILE):
        batch = request_payload(selected[start : start + FAMILIES_PER_FILE])
        assert_label_free(batch)
        write_json(args.output_dir / "requests" / "batches" / f"batch-{start // FAMILIES_PER_FILE + 1:02d}.json", batch)
    selection_labels = Counter(family["historic_label"] for family in selected)
    metadata = {
        "schema_version": 1,
        "source_dataset": args.dataset,
        "source_revision": args.revision,
        "source_config": args.config,
        "source_split": args.split,
        "source_sha256": source_digest(rows),
        "source_verification_sha256": verification_digest,
        "main_revision_sha_before": main_sha_before,
        "main_revision_sha_after": main_sha_after,
        **summary,
        "study_families": len(selected),
        "pilot_families": PILOT_FAMILIES,
        "variants_per_family_requested": VARIANTS_PER_FAMILY,
        "study_variants_requested": len(selected) * VARIANTS_PER_FAMILY,
        "pilot_variants_requested": PILOT_FAMILIES * VARIANTS_PER_FAMILY,
        "source_row_historic_label_counts": {
            "false": sum(row["model_is_robust"] is False for row in rows),
            "true": sum(row["model_is_robust"] is True for row in rows),
        },
        "family_historic_label_counts": {
            "false": sum(family["historic_label"] is False for family in families),
            "true": sum(family["historic_label"] is True for family in families),
        },
        "study_historic_label_counts": {"false": selection_labels[False], "true": selection_labels[True]},
        "prompt_sha256": hashlib.sha256(GENERATOR_PROMPT.encode("utf-8")).hexdigest(),
        "input_sha256": {
            "all_original_tasks": payload_digest(all_payload),
            "study50": payload_digest(study_payload),
            **{
                f"batch_{start // FAMILIES_PER_FILE + 1:02d}": payload_digest(
                    request_payload(selected[start : start + FAMILIES_PER_FILE])
                )
                for start in range(0, len(selected), FAMILIES_PER_FILE)
            },
        },
    }
    write_json(args.output_dir / ".source-metadata.json", metadata)
    write_json(
        args.output_dir / ".source-mapping.json",
        {
            "family_key": "dataset_id + problem_id",
            "deduplication": "one family per existing dataset_id/problem_id pair; no text or label conflicts accepted",
            "families": [
                {
                    "family_id": family["family_id"],
                    "dataset_id": family["dataset_id"],
                    "problem_id": family["problem_id"],
                    "historic_label": family["historic_label"],
                    "source_rows": family["source_rows"],
                    "model_id": family["model_id"],
                }
                for family in deterministic_order(families)
            ],
        },
    )
    write_json(
        args.output_dir / ".source-rows.json",
        {
            "rows": [
                {
                    "source_row_index": index,
                    "model_id": required_string(row, "model_id", index + 1),
                    "dataset_id": required_string(row, "dataset_id", index + 1),
                    "problem_id": required_string(row, "problem_id", index + 1),
                    "original_problem": required_string(row, "original_problem", index + 1),
                    "model_is_robust": row["model_is_robust"],
                    "permutation_type": row.get("permutation_type"),
                }
                for index, row in enumerate(rows)
            ]
        },
    )
    write_index(args.output_dir, all_count=len(families), study_count=len(selected))
    atomic_write(args.output_dir / "prompt.md", GENERATOR_PROMPT)
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
