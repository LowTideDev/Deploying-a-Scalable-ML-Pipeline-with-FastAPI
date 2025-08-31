import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference


cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture
def sample_data():
    """Load and process a small subset of census data for testing."""
    data = pd.read_csv("data/census.csv").sample(100, random_state=0)
    X, y, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    return X, y, encoder, lb


def test_train_model_returns_random_forest(sample_data):
    """The training function should return a RandomForestClassifier."""
    X, y, _, _ = sample_data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


def test_compute_model_metrics_expected_values():
    """Metrics should match expected precision, recall, and f1."""
    y_true = [0, 1, 1, 0]
    preds = [0, 1, 0, 0]
    precision, recall, f1 = compute_model_metrics(y_true, preds)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)
    assert f1 == pytest.approx(2 * (1.0 * 0.5) / (1.0 + 0.5))


def test_inference_matches_label_shape(sample_data):
    """Inference results should have the same length as the input labels."""
    X, y, _, _ = sample_data
    model = train_model(X, y)
    preds = inference(model, X)
    assert preds.shape == y.shape
