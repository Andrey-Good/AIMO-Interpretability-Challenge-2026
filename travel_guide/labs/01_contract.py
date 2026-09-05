"""Три искусственные задачи проходят через НАСТОЯЩИЕ ingestion и scoring.

Только стандартная библиотека. Ни сети, ни LLM, ни настоящих меток.
Запуск из корня: python travel_guide/labs/01_contract.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from _repo import ROOT


def run_variant(root: Path, name: str, submission: Path) -> tuple[list[dict], dict]:
    scoring_input = root / name
    result_dir = scoring_input / "res"
    reference_dir = scoring_input / "ref"
    reference_dir.mkdir(parents=True)
    labels = [{"id": "a1", "is_robust": True}, {"id": "b1", "is_robust": True}, {"id": "a2", "is_robust": False}]
    (reference_dir / "labels.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in labels), encoding="utf-8"
    )
    # Два отдельных процесса, как в scripts/run_local.py. -B не создаёт pycache.
    subprocess.run([
        sys.executable, "-B", str(ROOT / "components/ingestion_program/ingestion.py"),
        str(root / "input"), str(result_dir), str(submission),
    ], check=True)
    subprocess.run([
        sys.executable, "-B", str(ROOT / "components/scoring_program/scoring.py"),
        str(scoring_input), str(root / name / "scores"),
    ], check=True)
    predictions = [json.loads(line) for line in (result_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    scores = json.loads((root / name / "scores/scores.json").read_text(encoding="utf-8"))
    print(f"\n{name}: predictions = {predictions}")
    print(f"{name}: scores = {scores}")
    return predictions, scores


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="aimo-guide-") as temporary:
        root = Path(temporary)
        (root / "input").mkdir()
        # Специально перемешиваем модели: A1, B1, A2.
        cases = [
            {"id": "a1", "model_id": "toy/model-a", "problem": "First toy problem"},
            {"id": "b1", "model_id": "toy/model-b", "problem": "Second toy problem"},
            {"id": "a2", "model_id": "toy/model-a", "problem": "Third toy problem"},
        ]
        (root / "input/cases.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in cases), encoding="utf-8"
        )
        print("Входные ID:", [row["id"] for row in cases])
        predictions, scores = run_variant(root, "always-true", ROOT / "solutions/always-true")
        assert [row["id"] for row in predictions] == ["a1", "b1", "a2"]
        assert scores == {"accuracy": 2 / 3, "coverage": 1.0, "invalid_predictions": 0}

        broken = root / "broken-solution"
        broken.mkdir()
        (broken / "solution.py").write_text(
            "def are_robust(model_id: str, problems: list[str]) -> list[bool]:\n"
            "    return [True, 0] if model_id.endswith('-a') else [True]\n", encoding="utf-8"
        )
        predictions, scores = run_variant(root, "one-wrong-type", broken)
        assert [row["valid"] for row in predictions] == [False, True, False]
        assert scores == {"accuracy": 1 / 3, "coverage": 1 / 3, "invalid_predictions": 2}
        print("\nОдин int испортил весь вызов model-a, но не затронул model-b.")
        print("OK. Эти оценки проверяют контракт, а не качество AIMO-метода.")


if __name__ == "__main__":
    main()
