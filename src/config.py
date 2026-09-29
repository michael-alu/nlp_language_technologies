from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIRECTORY = ROOT / "data" / "raw"
PROCESSED_DATA_DIRECTORY = ROOT / "data" / "processed"
FIGURES_DIRECTORY = ROOT / "results" / "figures"

RAW_TRAIN_FILE = RAW_DATA_DIRECTORY / "Train.csv"
RAW_TEST_FILE = RAW_DATA_DIRECTORY / "Test.csv"

TRAIN_FILE = PROCESSED_DATA_DIRECTORY / "train.csv"
VALIDATION_FILE = PROCESSED_DATA_DIRECTORY / "validation.csv"
TEST_FILE = PROCESSED_DATA_DIRECTORY / "test.csv"
ZINDI_TEST_FILE = PROCESSED_DATA_DIRECTORY / "zindi_test.csv"

RANDOM_SEED = 42

VALIDATION_SHARE = 0.15
TEST_SHARE = 0.15

MINIMUM_WORDS_PER_ARTICLE = 5

LABELS = ["kitaifa", "michezo", "biashara", "kimataifa", "burudani"]

LABEL_MEANINGS = {
    "kitaifa": "national",
    "michezo": "sports",
    "biashara": "business",
    "kimataifa": "international",
    "burudani": "entertainment",
}
