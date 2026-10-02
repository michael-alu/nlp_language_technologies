from pathlib import Path

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc, confusion_matrix, roc_curve

from src import config
from src.evaluation import get_predicted_ids


def get_model_figures_directory(model_name: str) -> Path:
    model_directory = config.FIGURES_DIRECTORY / model_name

    model_directory.mkdir(parents=True, exist_ok=True)

    return model_directory


def plot_confusion_matrix(
    true_ids: np.ndarray,
    probabilities: np.ndarray,
    model_name: str,
    split_name: str = "validation",
) -> None:
    predicted_ids = get_predicted_ids(probabilities)

    all_label_ids = list(range(len(config.LABELS)))

    matrix = confusion_matrix(true_ids, predicted_ids, labels=all_label_ids)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=config.LABELS,
        yticklabels=config.LABELS,
    )

    plt.title(f"Confusion matrix: {model_name} ({split_name})")

    plt.xlabel("Predicted label")

    plt.ylabel("True label")

    plt.tight_layout()

    file_path = (
        get_model_figures_directory(model_name) / f"confusion_matrix_{split_name}.png"
    )

    plt.savefig(file_path, dpi=150)

    plt.show()


def plot_roc_curves(
    true_ids: np.ndarray,
    probabilities: np.ndarray,
    model_name: str,
    split_name: str = "validation",
) -> None:
    plt.figure(figsize=(6, 5))

    for label_id, label in enumerate(config.LABELS):
        is_this_class = true_ids == label_id

        if is_this_class.sum() == 0:
            continue

        class_probabilities = probabilities[:, label_id]

        false_positive_rate, true_positive_rate, _ = roc_curve(
            is_this_class, class_probabilities
        )

        area_under_curve = auc(false_positive_rate, true_positive_rate)

        plt.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{label} (AUC = {area_under_curve:.2f})",
        )

    plt.plot([0, 1], [0, 1], color="grey", linestyle="--")

    plt.title(f"ROC curves (one class vs rest): {model_name}")

    plt.xlabel("False positive rate")

    plt.ylabel("True positive rate")

    plt.legend()

    plt.tight_layout()

    file_path = get_model_figures_directory(model_name) / f"roc_curves_{split_name}.png"

    plt.savefig(file_path, dpi=150)

    plt.show()


def plot_learning_curves(
    history: dict[str, list[float]], model_name: str, experiment_name: str = ""
) -> None:
    # One value per epoch, for example
    # {"train_loss": [...], "validation_loss": [...], "validation_macro_f1": [...]}
    epochs = range(1, len(history["train_loss"]) + 1)

    figure, (loss_axis, f1_axis) = plt.subplots(1, 2, figsize=(11, 4))

    loss_axis.plot(epochs, history["train_loss"], label="train")

    loss_axis.plot(epochs, history["validation_loss"], label="validation")

    loss_axis.set_title("Loss per epoch")

    loss_axis.set_xlabel("Epoch")

    loss_axis.set_ylabel("Loss")

    loss_axis.legend()

    if "validation_macro_f1" in history:
        f1_axis.plot(epochs, history["validation_macro_f1"], color="green")

    f1_axis.set_title("Validation macro-F1 per epoch")

    f1_axis.set_xlabel("Epoch")

    f1_axis.set_ylabel("Macro-F1")

    figure.suptitle(f"Learning curves: {model_name} {experiment_name}")

    plt.tight_layout()

    file_name = "learning_curves.png"

    if experiment_name:
        file_name = f"learning_curves_{experiment_name}.png"

    file_path = get_model_figures_directory(model_name) / file_name

    plt.savefig(file_path, dpi=150)

    plt.show()
