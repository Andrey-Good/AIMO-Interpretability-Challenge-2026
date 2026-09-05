"""CPU-only tests for the guide and selected original numerical functions.

No model downloads, no pickle/joblib loading, no changes to competition code.
"""
from __future__ import annotations

import ast
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as functional

GUIDE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GUIDE / "labs"))
sys.path.insert(0, str(GUIDE / "tools"))
from _repo import ROOT, load_functions, load_module
from check_guide import check_links, check_snapshots, check_syntax, find_symbol
from normalize_cases import normalize_file, normalize_lines

PREFIX = "solutions/uncertainty-profiling/uncertainty_profile/"
METRICS = load_module(PREFIX + "metrics.py", "guide_test_metrics")
INGESTION = load_module("components/ingestion_program/ingestion.py", "guide_test_ingestion")
SCORING = load_module("components/scoring_program/scoring.py", "guide_test_scoring")
PROBE = load_functions("solutions/trained-probe/probe_inference.py",
                       ["_layer_index", "mean_ensemble_margin"], {"np": np})


def make_case(identifier="a", problem="text"):
    return {"id": identifier, "model_id": "toy/model", "problem": problem}


def metric_row(probabilities, *, fraction=0.2):
    probs = np.asarray(probabilities, dtype=float)
    return METRICS.compute_generation_confidence_metrics(
        log_probs=np.log(probs), probs=probs, entropy=np.zeros_like(probs),
        top1_probs=probs, top2_margins=np.zeros_like(probs),
        selected_is_top1=np.ones_like(probs, dtype=bool), min_k_fraction=fraction,
        high_conf_threshold=0.9, low_conf_threshold=0.1)


def artifact_for(groups):
    return SimpleNamespace(kind="pickle", data={"best_layer_index": 2, "groups": groups})


def probe_group(biases, key=2):
    return {"probes": {key: {"weights": np.zeros((len(biases), 3)),
                            "bias": biases, "threshold": np.zeros(len(biases))}}}


class NormalizationTests(unittest.TestCase):
    def test_preserves_order_ids_and_other_metadata(self):
        old = make_case("a", {"original_problem": "текст", "permutation_type": ["x"]})
        new = make_case("b", "another")
        rows, changed = normalize_lines(map(json.dumps, [old, new]))
        self.assertEqual(rows, [make_case("a", "текст"), new])
        self.assertEqual(changed, 1)
        self.assertIsInstance(old["problem"], dict)

    def test_rejects_bad_rows(self):
        bad = [[], ["null"], ["{"], [json.dumps(make_case(problem=""))],
               [json.dumps(make_case(problem={"original_problem": 3}))],
               [json.dumps(make_case()), json.dumps(make_case())]]
        for lines in bad:
            with self.subTest(lines=lines), self.assertRaises(ValueError):
                normalize_lines(lines)

    def test_file_source_untouched_and_ingestion_accepts_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, target = Path(temporary) / "old", Path(temporary) / "new"
            text = json.dumps(make_case(problem={"original_problem": "text"})) + "\n"
            source.write_text(text, encoding="utf-8")
            self.assertEqual(normalize_file(source, target), (1, 1))
            self.assertEqual(source.read_text(encoding="utf-8"), text)
            self.assertEqual(INGESTION.load_cases(target), [make_case()])

    def test_rejects_overwriting_and_same_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, target = Path(temporary) / "source", Path(temporary) / "target"
            source.write_text(json.dumps(make_case()), encoding="utf-8")
            target.write_text("keep", encoding="utf-8")
            with self.assertRaises(ValueError):
                normalize_file(source, source)
            with self.assertRaises(FileExistsError):
                normalize_file(source, target)
            self.assertEqual(target.read_text(encoding="utf-8"), "keep")

    def test_invalid_source_creates_no_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, target = Path(temporary) / "source", Path(temporary) / "target"
            source.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                normalize_file(source, target)
            self.assertFalse(target.exists())


class ProbeTests(unittest.TestCase):
    def test_all_seeds_not_mean_of_groups(self):
        artifact = artifact_for([probe_group([3]), probe_group([-3, -3, -3])])
        margin = PROBE["mean_ensemble_margin"](np.zeros(3), artifact)
        self.assertEqual(margin, -1.5)  # mean of group means would incorrectly be 0.

    def test_accepts_string_layer_key_and_zero_margin(self):
        margin = PROBE["mean_ensemble_margin"](np.zeros(3), artifact_for([probe_group([0], "2")]))
        self.assertEqual(margin, 0)
        self.assertTrue(bool(margin >= 0))

    def test_missing_selected_layer_fails(self):
        with self.assertRaisesRegex(RuntimeError, "no probes"):
            PROBE["mean_ensemble_margin"](np.zeros(3), artifact_for([probe_group([1], 7)]))

    def test_bad_dimension_fails(self):
        with self.assertRaisesRegex(RuntimeError, "dimension mismatch"):
            PROBE["mean_ensemble_margin"](np.zeros(4), artifact_for([probe_group([1])]))

    def test_strategy_layer_overrides_best(self):
        artifact = artifact_for([])
        artifact.data["recommended_strategy"] = {"layer_index": 4}
        self.assertEqual(PROBE["_layer_index"](artifact), 4)


class MetricTests(unittest.TestCase):
    def test_known_nll_and_perplexity(self):
        row = metric_row([0.7, 0.4])
        self.assertAlmostEqual(row["generation_ppl"], 1 / math.sqrt(0.7 * 0.4))
        self.assertAlmostEqual(row["generation_mean_nll"], -math.log(0.7 * 0.4) / 2)

    def test_min_k_rounds_up_and_sorts(self):
        row = metric_row([0.9, 0.2, 0.5], fraction=0.5)
        self.assertAlmostEqual(row["generation_min_k_logprob"], math.log(0.2 * 0.5) / 2)

    def test_inclusive_confidence_thresholds(self):
        row = metric_row([0.1, 0.9])
        self.assertEqual(row["generation_frac_high_conf_tokens"], 0.5)
        self.assertEqual(row["generation_frac_low_conf_tokens"], 0.5)

    def test_empty_tokens_are_not_zero_uncertainty(self):
        row = metric_row([])
        self.assertEqual(row["generation_num_tokens"], 0)
        self.assertTrue(math.isnan(row["generation_mean_entropy"]))

    def test_std_is_population_std(self):
        row = metric_row([0.2, 0.8])
        self.assertAlmostEqual(row["generation_std_logprob"], np.std(np.log([0.2, 0.8]), ddof=0))

    def test_invalid_min_k(self):
        with self.assertRaises(ValueError):
            metric_row([0.5], fraction=0)


class CollectorTests(unittest.TestCase):
    def test_pending_alignment_finalization_and_eos(self):
        # Extract two actual methods without importing Transformers or loading a model.
        path = ROOT / PREFIX / "extraction.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        methods = [find_symbol(tree, f"GenerationConfidenceCollector.{name}")
                   for name in ("__call__", "compute_features")]
        module = ast.Module(body=[ast.parse("from __future__ import annotations").body[0], *methods], type_ignores=[])

        def mask_tokens(ids, *, pad_token_id, eos_token_ids):
            mask = torch.ones_like(ids, dtype=torch.bool)
            for token in eos_token_ids | ({pad_token_id} if pad_token_id is not None else set()):
                mask &= ids != token
            return mask

        scope = {"torch": torch, "np": np, "functional": functional,
                 "compute_generation_confidence_metrics": METRICS.compute_generation_confidence_metrics,
                 "build_valid_token_mask": mask_tokens}
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
        collector = SimpleNamespace(pending_log_probs=None, selected_log_probs=[], entropy=[],
                                    top1_probs=[], top2_margins=[], top1_token_ids=[])
        first = torch.tensor([[0.6, 0.3, 0.1]]).log()
        second = torch.tensor([[0.1, 0.2, 0.7]]).log()
        self.assertIs(scope["__call__"](collector, torch.tensor([[8]]), first), first)
        self.assertEqual(len(collector.selected_log_probs), 0)
        scope["__call__"](collector, torch.tensor([[8, 0]]), second)
        self.assertEqual(len(collector.selected_log_probs), 1)
        self.assertAlmostEqual(collector.selected_log_probs[0].item(), math.log(0.6), places=6)
        rows, ids = scope["compute_features"](
            collector, sequences=torch.tensor([[8, 0, 2]]), prompt_length=1,
            pad_token_id=2, eos_token_ids={2}, config=SimpleNamespace(
                min_k_fraction=0.2, high_conf_threshold=0.9, low_conf_threshold=0.1))
        self.assertEqual(ids.tolist(), [[0, 2]])
        self.assertEqual(len(collector.selected_log_probs), 2)
        self.assertIsNone(collector.pending_log_probs)
        self.assertEqual(rows[0]["generation_num_tokens"], 1)
        self.assertAlmostEqual(rows[0]["generation_mean_logprob"], math.log(0.6), places=6)


class ScoringAndGuideTests(unittest.TestCase):
    def test_accuracy_denominator_includes_invalid(self):
        result = SCORING.compute_scores({"a": True, "b": False},
                 {"a": {"valid": True, "is_robust": True}}, set(), 0)
        self.assertEqual(result, {"accuracy": 0.5, "coverage": 0.5, "invalid_predictions": 1})

    def test_structural_errors_are_counted(self):
        result = SCORING.compute_scores({"a": True},
                 {"a": {"valid": True, "is_robust": True}, "extra": {}}, {"a"}, 1)
        self.assertEqual(result["invalid_predictions"], 4)
        self.assertEqual(result["coverage"], 0)

    def test_snapshots_match_selected_source_functions(self):
        self.assertEqual(check_snapshots(), [])

    def test_python_syntax(self):
        self.assertEqual(check_syntax(), [])

    def test_link_checker_detects_missing_local_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            guide = Path(temporary) / "guide"
            guide.mkdir()
            (guide / "README.md").write_text("[bad](missing.md) [external](https://example.com)", encoding="utf-8")
            self.assertEqual(len(check_links(guide)), 1)


if __name__ == "__main__":
    unittest.main()
