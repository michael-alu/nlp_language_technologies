import ast
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def unwrap_list_text(text):
    # Some sports articles were saved as a Python list of paragraphs,
    # for example "['First paragraph. ', 'Second paragraph.']"
    looks_like_a_list = text.startswith("[") and text.endswith("]")
    if not looks_like_a_list:
        return text

    try:
        paragraphs = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text.strip("[]")

    if not isinstance(paragraphs, list):
        return text

    paragraphs_as_text = [str(paragraph) for paragraph in paragraphs]
    return " ".join(paragraphs_as_text)


def remove_urls(text):
    url_pattern = r"https?://\S+|www\.\S+"
    return re.sub(url_pattern, " ", text)


def replace_fancy_characters(text):
    replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "…": "...",
        "–": "-",
        " ": " ",
        "ﬁ": "fi",
    }
    for fancy_character, plain_character in replacements.items():
        text = text.replace(fancy_character, plain_character)
    return text


def add_space_after_sentence_end(text):
    # Many sentences are glued together, like "yoyote.Hayo yalisemwa"
    glued_sentence_pattern = r"([.!?])([A-Za-z])"
    return re.sub(glued_sentence_pattern, r"\1 \2", text)


def collapse_whitespace(text):
    single_spaced_text = re.sub(r"\s+", " ", text)
    return single_spaced_text.strip()


def clean_text(text):
    text = str(text).strip()
    text = unwrap_list_text(text)
    text = remove_urls(text)
    text = replace_fancy_characters(text)
    text = add_space_after_sentence_end(text)
    text = collapse_whitespace(text)
    return text


def count_words(text):
    return len(text.split())


def clean_labels(labels):
    return labels.str.strip().str.lower()


def remove_near_empty_articles(data):
    word_counts = data["content"].apply(count_words)
    has_enough_words = word_counts >= config.MINIMUM_WORDS_PER_ARTICLE

    removed_rows = data[~has_enough_words]
    print(f"Removing {len(removed_rows)} near-empty articles:")
    print(removed_rows[["id", "category", "content"]].to_string())

    return data[has_enough_words].reset_index(drop=True)


def clean_train_data(raw_train):
    train = raw_train.copy()
    train["content"] = train["content"].apply(clean_text)
    train["category"] = clean_labels(train["category"])
    train = remove_near_empty_articles(train)
    return train


def clean_zindi_test_data(raw_test):
    test = raw_test.copy()
    test = test.rename(columns={"swahili_id": "id"})
    test["content"] = test["content"].apply(clean_text)
    return test


def split_train_validation_test(data):
    # Stratified splits keep the rare classes (burudani has only 17 articles)
    # in every split, with the same class proportions as the full data
    holdout_share = config.VALIDATION_SHARE + config.TEST_SHARE

    train, holdout = train_test_split(
        data,
        test_size=holdout_share,
        stratify=data["category"],
        random_state=config.RANDOM_SEED,
    )

    test_share_of_holdout = config.TEST_SHARE / holdout_share

    validation, test = train_test_split(
        holdout,
        test_size=test_share_of_holdout,
        stratify=holdout["category"],
        random_state=config.RANDOM_SEED,
    )

    return train, validation, test


def print_split_summary(train, validation, test):
    summary = pd.DataFrame(
        {
            "train": train["category"].value_counts(),
            "validation": validation["category"].value_counts(),
            "test": test["category"].value_counts(),
        }
    )
    summary.loc["total"] = summary.sum()
    print(summary)


def main():
    raw_train = pd.read_csv(config.RAW_TRAIN_FILE)
    raw_test = pd.read_csv(config.RAW_TEST_FILE)

    clean_train = clean_train_data(raw_train)
    clean_zindi_test = clean_zindi_test_data(raw_test)

    train, validation, test = split_train_validation_test(clean_train)
    print_split_summary(train, validation, test)

    config.PROCESSED_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    train.to_csv(config.TRAIN_FILE, index=False)
    validation.to_csv(config.VALIDATION_FILE, index=False)
    test.to_csv(config.TEST_FILE, index=False)
    clean_zindi_test.to_csv(config.ZINDI_TEST_FILE, index=False)

    print(f"Saved cleaned files to {config.PROCESSED_DATA_DIRECTORY}")


if __name__ == "__main__":
    main()
