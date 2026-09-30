import numpy as np
from sklearn.metrics import log_loss

from src import config


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled_logits = logits / temperature

    # Subtracting the largest value stops np.exp from overflowing on big numbers.
    # It does not change the result, because softmax only depends on the differences.
    largest_logit_per_row = scaled_logits.max(axis=1, keepdims=True)

    shifted_logits = scaled_logits - largest_logit_per_row

    exponentials = np.exp(shifted_logits)

    return exponentials / exponentials.sum(axis=1, keepdims=True)


def find_best_temperature(
    validation_logits: np.ndarray, validation_ids: np.ndarray
) -> float:
    # Neural networks are often overconfident (Guo et al., 2017).
    # We try many temperatures and keep the one with the lowest validation log loss.

    all_label_ids = list(range(len(config.LABELS)))

    temperatures_to_try = np.arange(0.5, 5.01, 0.05)

    best_temperature = 1.0

    best_log_loss = float("inf")

    for temperature in temperatures_to_try:
        probabilities = softmax(validation_logits, temperature)

        current_log_loss = log_loss(validation_ids, probabilities, labels=all_label_ids)

        if current_log_loss < best_log_loss:
            best_log_loss = current_log_loss

            best_temperature = float(temperature)

    print(
        f"Best temperature: {best_temperature:.2f} (validation log loss {best_log_loss:.4f})"
    )

    return best_temperature
