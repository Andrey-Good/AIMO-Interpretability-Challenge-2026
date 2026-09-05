# Учебная выдержка _run; оригинал: components/ingestion_program/ingestion.py
# Помощники load_cases/load_solution находятся в оригинале.
from __future__ import annotations

if __name__ == "__main__":
    raise SystemExit("Это выдержка для чтения. Запускай travel_guide/labs/01_contract.py")


def _run(input_dir: Path, output_dir: Path, submission_dir: Path) -> None:
    cases = load_cases(input_dir / "cases.jsonl")
    solution = load_solution(submission_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # GUIDE: один вызов на model_id, не один вызов на строку.
    # Храним исходные индексы, чтобы затем восстановить порядок.
    batches: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, case in enumerate(cases):
        batches.setdefault(case["model_id"], []).append((index, case))

    predictions = [
        {"id": case["id"], "is_robust": False, "valid": False} for case in cases
    ]
    failures = 0
    for model_id, batch in batches.items():
        batch_predictions = [False] * len(batch)
        batch_valid = [False] * len(batch)
        try:
            problems = [case["problem"] for _, case in batch]
            results = solution.are_robust(model_id, problems)
            # GUIDE: один неподходящий элемент делает невалидным весь batch.
            # Принимается именно Python bool, не int и не np.bool_.
            if (
                type(results) is list
                and len(results) == len(problems)
                and all(type(result) is bool for result in results)
            ):
                batch_predictions = results
                batch_valid = [True] * len(batch)
        # GUIDE: перехватываются ошибки вызова метода, не ранней загрузки cases.
        except Exception:
            # Participant exception details are intentionally not copied to results.
            pass

        failures += batch_valid.count(False)
        for (index, case), prediction, valid in zip(batch, batch_predictions, batch_valid):
            predictions[index] = {
                "id": case["id"],
                "is_robust": prediction,
                "valid": valid,
            }

    # GUIDE: False/valid=True — прогноз; False/valid=False — ошибка обработки.
    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as stream:
        for prediction in predictions:
            stream.write(json.dumps(prediction, sort_keys=True, separators=(",", ":")))
            stream.write("\n")

    summary = {"cases": len(cases), "invalid_predictions": failures}
    (output_dir / "ingestion_summary.json").write_text(
        json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
    )
