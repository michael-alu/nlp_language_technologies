from typing import Callable

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from src import config
from src import data_loading
from src.evaluation import compute_scores, summarise_scores

NUMBER_OF_FOLDS = 5

NUMBER_OF_REPEATS = 3


def cross_validate(build_model: Callable, data: pd.DataFrame) -> dict[str, float]:
    # build_model must return a new, untrained model with fit() and predict_proba(),
    # for example a scikit-learn Pipeline. A new model is built for every fold.

    texts = data["content"]

    label_ids = data_loading.labels_to_ids(data["category"])

    folds = RepeatedStratifiedKFold(
        n_splits=NUMBER_OF_FOLDS,
        n_repeats=NUMBER_OF_REPEATS,
        random_state=config.RANDOM_SEED,
    )

    scores_per_fold = []

    fold_splits = folds.split(texts, label_ids)

    for fold_number, (train_rows, holdout_rows) in enumerate(fold_splits, start=1):
        model = build_model()

        model.fit(texts.iloc[train_rows], label_ids[train_rows])

        holdout_probabilities = model.predict_proba(texts.iloc[holdout_rows])

        fold_scores = compute_scores(label_ids[holdout_rows], holdout_probabilities)

        scores_per_fold.append(fold_scores)

        print(f"===Fold {fold_number}: macro-F1 = {fold_scores['macro_f1']:.4f}===")

    summary = summarise_scores(scores_per_fold)

    print(
        f"Mean macro-F1 = {summary['macro_f1']:.4f} (std {summary['macro_f1_std']:.4f})"
    )

    return summary
