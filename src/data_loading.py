import pandas as pd

from src import config


def load_train():
    return pd.read_csv(config.TRAIN_FILE)


def load_validation():
    return pd.read_csv(config.VALIDATION_FILE)


def load_test():
    return pd.read_csv(config.TEST_FILE)


def load_zindi_test():
    return pd.read_csv(config.ZINDI_TEST_FILE)


def labels_to_ids(labels):
    label_to_id = {label: index for index, label in enumerate(config.LABELS)}
    return labels.map(label_to_id).to_numpy()


def ids_to_labels(ids):
    return [config.LABELS[index] for index in ids]
