import numpy as np
import pandas as pd

from src.train_baselines import build_preprocessor, evaluate


def test_evaluate_returns_expected_confusion_counts() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.3, 0.9])

    metrics = evaluate(y_true, probabilities, threshold=0.5)

    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert metrics["recall"] == 0.5
    assert metrics["precision"] == 0.5


def test_preprocessor_handles_unseen_validation_category() -> None:
    train = pd.DataFrame({"amount": [1.0, np.nan, 3.0], "segment": ["A", "B", "A"]})
    validation = pd.DataFrame({"amount": [2.0], "segment": ["unseen"]})
    preprocessor = build_preprocessor(["amount"], ["segment"])

    train_transformed = preprocessor.fit_transform(train)
    validation_transformed = preprocessor.transform(validation)

    assert train_transformed.shape[1] == validation_transformed.shape[1]
    values = (
        validation_transformed.toarray()
        if hasattr(validation_transformed, "toarray")
        else validation_transformed
    )
    assert np.isfinite(values).all()
