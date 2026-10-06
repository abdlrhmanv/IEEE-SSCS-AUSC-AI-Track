"""Regression tests for data isolation and categorical preprocessing."""
from pathlib import Path
import inspect
import json
import hashlib
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import choose_setting, load_split, pipeline, preprocessing, tune

TASK = Path(__file__).resolve().parents[1]


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split = load_split(TASK / "mushrooms.csv")
        split = cls.split
        train_ids = np.concatenate([split.y_train[split.y_train == c].index[:20] for c in [0, 1]])
        val_ids = np.concatenate([split.y_validation[split.y_validation == c].index[:20] for c in [0, 1]])
        cls.X = split.X_train.loc[train_ids]
        cls.y = split.y_train.loc[train_ids]
        cls.V = split.X_validation.loc[val_ids]
        cls.v = split.y_validation.loc[val_ids]
        cls.grids = {"knn": ("K", [2, 5]), "sgd": ("learning_rate", [0.001, 0.01]), "logreg": ("max_iter", [1000])}

    def test_disjoint_deterministic_stratified_partitions(self):
        split = self.split
        sets = [set(frame.index) for frame in [split.X_train, split.X_validation, split.X_test]]
        self.assertEqual([len(s) for s in sets], [4874, 1625, 1625])
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(len(set.union(*sets)), 8124)
        again = load_split(TASK / "mushrooms.csv")
        self.assertTrue(split.X_train.index.equals(again.X_train.index))
        for y in [split.y_train, split.y_validation, split.y_test]:
            self.assertLess(abs(float(y.mean()) - 3916 / 8124), 0.002)
        self.assertFalse((split.X_train == "?").any().any())

    def test_selection_ignores_test_metrics_and_breaks_ties_by_cost(self):
        table = pd.DataFrame({"K": [10, 2, 5], "validation_F1": [0.8, 0.9, 0.9], "test_F1": [1.0, 0.0, 0.5]})
        self.assertEqual(choose_setting(table, "K"), 2)
        table["test_F1"] = [0.0, 1.0, 1.0]
        self.assertEqual(choose_setting(table, "K"), 2)
        self.assertNotIn("X_test", inspect.signature(tune).parameters)
        self.assertNotIn("y_test", inspect.signature(tune).parameters)

    def test_validation_only_category_does_not_enter_fitted_encoder(self):
        train = pd.DataFrame({"color": ["red", "blue", "red"]})
        heldout = pd.DataFrame({"color": ["never-seen-before"]})
        for encoding in ["ordinal", "onehot"]:
            with self.subTest(encoding=encoding):
                transform = preprocessing(encoding)
                transform.fit(train)
                categories = transform.named_steps["encoder"].categories_[0]
                self.assertNotIn("never-seen-before", categories)
                result = transform.transform(heldout)
                self.assertTrue(np.isfinite(result).all())
                self.assertEqual(result.shape[1], transform.transform(train).shape[1])

    def test_cached_sweeps_match_full_pipelines(self):
        with threadpool_limits(limits=1):
            for encoding in ["ordinal", "onehot"]:
                tables, settings, _ = tune(self.X, self.y, self.V, self.v, encoding, self.grids)
                for kind, (parameter, values) in self.grids.items():
                    for value in values:
                        model = pipeline(encoding, kind, value).fit(self.X, self.y)
                        row = tables[kind].loc[tables[kind][parameter] == value].iloc[0]
                        self.assertAlmostEqual(row["train_F1"], f1_score(self.y, model.predict(self.X)))
                        self.assertAlmostEqual(row["validation_F1"], f1_score(self.v, model.predict(self.V)))
                    self.assertEqual(settings[kind]["value"], choose_setting(tables[kind], parameter))

    def test_named_csv_preserves_raw_dataset_with_missing_value_serialization(self):
        raw = pd.read_csv(TASK / "mushrooms_raw.data", header=None, dtype=str)
        named = pd.read_csv(TASK / "mushrooms.csv", dtype=str)
        self.assertEqual(int(named.isna().sum().sum()), 2480)
        np.testing.assert_array_equal(raw.to_numpy(), named.fillna("?").to_numpy())

    def test_recorded_grids_and_choices_match_protocol(self):
        protocol = json.loads((TASK / "results/evaluation_protocol.json").read_text())
        self.assertEqual(protocol["dataset_sha256"], hashlib.sha256((TASK / "mushrooms.csv").read_bytes()).hexdigest())
        filenames = {"knn": "knn_f1_by_k", "sgd": "logreg_f1_by_lr", "logreg": "logreg_f1_by_iter"}
        count = 0
        for encoding in ["ordinal", "onehot"]:
            for kind, filename in filenames.items():
                table = pd.read_csv(TASK / "results" / f"{encoding}_{filename}.csv")
                parameter, values = __import__("evaluation").GRIDS[kind]
                np.testing.assert_allclose(table[parameter], values)
                self.assertNotIn("test_F1", table.columns)
                self.assertEqual(choose_setting(table, parameter), protocol["selections"][encoding][kind]["value"])
                count += len(table)
        self.assertEqual(count, 2498)
        final = pd.read_csv(TASK / "results/encoding_comparison.csv")
        self.assertEqual(len(final), 6)
        for row in final.itertuples():
            self.assertEqual(row.value, protocol["selections"][row.encoding][row.model]["value"])

    def test_invalid_validation_results_are_rejected(self):
        with self.assertRaises(ValueError):
            choose_setting(pd.DataFrame({"K": [], "validation_F1": []}), "K")
        with self.assertRaises(ValueError):
            choose_setting(pd.DataFrame({"K": [2], "validation_F1": [np.nan]}), "K")


if __name__ == "__main__":
    unittest.main()
