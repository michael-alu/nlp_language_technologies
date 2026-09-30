import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss

from src import config

RESULTS_DIRECTORY = config.ROOT / "results"
MISCLASSIFIED_DIRECTORY = RESULTS_DIRECTORY / "misclassified"
SUBMISSIONS_DIRECTORY = RESULTS_DIRECTORY / "submissions"


def get_predicted_ids(probabilities: np.ndarray) -> np.ndarray:
    return np.argmax(probabilities, axis=1)


def compute_scores(true_ids: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predicted_ids = get_predicted_ids(probabilities)

    all_label_ids = list(range(len(config.LABELS)))

    scores = {
        "accuracy": accuracy_score(true_ids, predicted_ids),
        "log_loss": log_loss(true_ids, probabilities, labels=all_label_ids),
        "macro_f1": f1_score(
            true_ids,
            predicted_ids,
            average="macro",
            labels=all_label_ids,
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            true_ids,
            predicted_ids,
            average="weighted",
            labels=all_label_ids,
            zero_division=0,
        ),
    }

    f1_per_class = f1_score(
        true_ids, predicted_ids, average=None, labels=all_label_ids, zero_division=0
    )

    for label, class_f1 in zip(config.LABELS, f1_per_class):
        scores[f"f1_{label}"] = class_f1

    for name, value in scores.items():
        scores[name] = round(float(value), 4)

    return scores


def print_classification_report(
    true_ids: np.ndarray, probabilities: np.ndarray
) -> None:
    predicted_ids = get_predicted_ids(probabilities)

    all_label_ids = list(range(len(config.LABELS)))

    report = classification_report(
        true_ids,
        predicted_ids,
        labels=all_label_ids,
        target_names=config.LABELS,
        zero_division=0,
    )

    print(report)


def save_misclassified_examples(
    data: pd.DataFrame,
    true_ids: np.ndarray,
    probabilities: np.ndarray,
    model_name: str,
) -> None:
    predicted_ids = get_predicted_ids(probabilities)

    is_wrong = predicted_ids != true_ids

    wrong_rows = data[is_wrong].copy()

    wrong_rows["true_label"] = [config.LABELS[index] for index in true_ids[is_wrong]]

    wrong_rows["predicted_label"] = [
        config.LABELS[index] for index in predicted_ids[is_wrong]
    ]

    wrong_rows["confidence"] = probabilities[is_wrong].max(axis=1).round(4)

    wrong_rows = wrong_rows.sort_values("confidence", ascending=False)

    MISCLASSIFIED_DIRECTORY.mkdir(parents=True, exist_ok=True)

    file_path = MISCLASSIFIED_DIRECTORY / f"{model_name}.csv"

    wrong_rows.to_csv(file_path, index=False)

    print(f"Saved {len(wrong_rows)} misclassified examples to {file_path}")


def save_zindi_submission(
    zindi_test: pd.DataFrame, probabilities: np.ndarray, model_name: str
) -> None:
    submission = pd.DataFrame(probabilities, columns=config.LABELS)

    submission.insert(0, "swahili_id", zindi_test["id"].values)

    SUBMISSIONS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    file_path = SUBMISSIONS_DIRECTORY / f"{model_name}.csv"

    submission.to_csv(file_path, index=False)

    print(f"Saved Zindi submission to {file_path}")


def summarise_scores(list_of_scores: list[dict[str, float]]) -> dict[str, float]:
    scores_table = pd.DataFrame(list_of_scores)

    summary = {}

    for name in scores_table.columns:
        summary[name] = round(float(scores_table[name].mean()), 4)

        summary[f"{name}_std"] = round(float(scores_table[name].std()), 4)

    return summary
