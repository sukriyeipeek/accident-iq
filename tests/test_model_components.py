import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.evaluation.classification import evaluate_predictions, ordered_probabilities, select_baseline
from src.evaluation.explainability import build_reference_row, local_sensitivity
from src.features.preprocessing import CATEGORICAL_FEATURES, RAW_FEATURES

pytest.importorskip("lightgbm")

from src.models.advanced import build_advanced, named_dense_frame
from src.models.baselines import BASELINE_NAMES, build_baseline

torch = pytest.importorskip("torch")
from src.models.tabular_nn import TabularMLP, class_weights


@pytest.fixture
def codebook():
    result = {field: {-1: "Data missing or out of range", 1: f"{field} one", 2: f"{field} two", 9: "Unknown"}
              for field in CATEGORICAL_FEATURES}
    result["junction_detail_historic"].update({0: "Outside junction", 3: "T junction"})
    result["junction_control"].update({0: "Outside junction", 4: "Give way"})
    return result


def test_ordered_probabilities_normalizes_float_roundoff_and_reorders_classes():
    actual = ordered_probabilities(np.array([[0.2, 0.3, 0.49999997]], dtype=np.float32), [2, 0, 1])
    np.testing.assert_allclose(actual, [[0.3, 0.49999997, 0.2]], atol=1e-7)
    assert actual.sum(axis=1)[0] == pytest.approx(1.0)


@pytest.mark.parametrize("probs,classes", [
    ([[0.5, 0.5]], [0, 0]), ([[0.2, 0.2]], [0, 3]), ([[float("nan"), 1]], [0, 1]),
    ([[0.2, 0.2]], [0, 1]), ([[0.2]], [0, 1]),
])
def test_invalid_probabilities_are_rejected(probs, classes):
    with pytest.raises(ValueError):
        ordered_probabilities(probs, classes)


def test_metrics_keep_all_three_labels_and_per_class_support():
    metrics = evaluate_predictions(np.array([0, 1, 2, 2]),
                                   np.array([[.8, .1, .1], [.2, .7, .1], [.1, .2, .7], [.1, .6, .3]]))
    assert metrics["accuracy"] == pytest.approx(.75)
    assert metrics["macro_f1"] > metrics["weighted_f1"]
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 1, 1]]
    assert metrics["per_class"]["Fatal"]["support"] == 1
    assert metrics["per_class"]["Slight"]["recall"] == .5


def test_baseline_selection_uses_mean_macro_f1_and_deterministic_tie_break():
    scores = {name: [{"metrics": {"macro_f1": 0.4}}, {"metrics": {"macro_f1": 0.2}}]
              for name in ["logistic", "dummy_prior"]}
    assert select_baseline(scores) == "dummy_prior"
    assert set(BASELINE_NAMES) == {"dummy_prior", "logistic", "logistic_balanced"}


def test_estimators_share_raw_input_preprocessor(codebook):
    baseline = build_baseline("logistic_balanced", codebook)
    protocol = json.loads((__import__("pathlib").Path(__file__).resolve().parents[1] /
                          "config/advanced_protocol.json").read_text(encoding="utf-8"))
    advanced = build_advanced("random_forest_balanced", codebook, protocol)
    assert tuple(baseline.named_steps) == ("preprocessing", "model")
    assert tuple(advanced.named_steps) == ("preprocessing", "model")
    assert advanced.named_steps["model"].n_estimators == 120


def test_lightgbm_named_dense_adapter_has_stable_columns():
    frame = named_dense_frame(sparse.csr_matrix([[0, 1], [2, 0]], dtype=np.float32))
    assert frame.columns.tolist() == ["Column_0", "Column_1"]
    np.testing.assert_array_equal(frame.to_numpy(), [[0, 1], [2, 0]])


def test_tabular_mlp_output_dimensions_and_balanced_weights():
    network = TabularMLP(63)
    output = network(torch.zeros((4, 63)))
    assert output.shape == (4, 3)
    assert class_weights(np.array([0, 0, 1, 2, 2, 2])).tolist() == pytest.approx([1.0, 2.0, 2 / 3])
    with pytest.raises(ValueError, match="All severity classes"):
        class_weights(np.array([0, 2]))


def test_reference_row_respects_structural_junction_state(codebook):
    records = pd.DataFrame({feature: [1, 1, 2, 2] for feature in CATEGORICAL_FEATURES})
    records["junction_detail_historic"] = [0, 0, 3, 3]
    records["junction_control"] = [4, 4, 4, 4]
    records["date"] = ["01/01/2019"] * 4
    records["time"] = ["12:00"] * 4
    records["speed_limit"] = [20, 30, 40, 50]
    reference = build_reference_row(records)
    if reference["junction_detail_historic"] == 0:
        assert reference["junction_control"] == 0


class SensitivityModel:
    classes_ = np.array([0, 1, 2])

    def predict_proba(self, X):
        shift = X["speed_limit"].fillna(0).to_numpy(dtype=float) / 1000
        return np.column_stack([0.1 + shift, np.full(len(X), 0.2), 0.7 - shift])


def test_local_sensitivity_is_ranked_and_non_causal(codebook):
    row = pd.DataFrame([{feature: 1 for feature in CATEGORICAL_FEATURES}])
    row["date"], row["time"], row["speed_limit"] = "2020-02-01", "20:00", 70
    reference = {feature: 1 for feature in CATEGORICAL_FEATURES}
    reference.update(date="01/07/2019", time="12:00", speed_limit=20)
    result = local_sensitivity(SensitivityModel(), row, reference, codebook, top_k=4)
    assert result["explained_class"] == "Slight"
    assert len(result["factors"]) == 4
    assert result["factors"][0]["feature"] == "speed_limit"
    assert "not causality" in result["interpretation"]
