import gc

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

from src import config
from src import data_loading
from src.calibration import softmax
from src.evaluation import compute_scores
from src.reproducibility import set_seed

MAX_TOKENS = 512

# Sun et al. (2019): keep the first 128 and the last 382 tokens of long articles
HEAD_TOKENS = 128


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def shorten_token_ids(token_ids: list[int], strategy: str) -> list[int]:
    room_for_article = MAX_TOKENS - 2

    if len(token_ids) <= room_for_article:
        return token_ids

    if strategy == "head":
        return token_ids[:room_for_article]

    tail_tokens = room_for_article - HEAD_TOKENS

    head = token_ids[:HEAD_TOKENS]

    tail = token_ids[-tail_tokens:]

    return head + tail


def encode_articles(
    tokenizer: PreTrainedTokenizerBase, data: pd.DataFrame, strategy: str
) -> list[dict]:
    # The Zindi test set has no labels, so we use 0 as a placeholder label
    if "category" in data.columns:
        label_ids = data_loading.labels_to_ids(data["category"])
    else:
        label_ids = np.zeros(len(data), dtype=int)

    encoded_articles = []

    for text, label_id in zip(data["content"], label_ids):
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

        short_token_ids = shorten_token_ids(token_ids, strategy)

        input_ids = (
            [tokenizer.cls_token_id] + short_token_ids + [tokenizer.sep_token_id]
        )

        encoded_articles.append({"input_ids": input_ids, "labels": int(label_id)})

    return encoded_articles


def build_data_loader(
    encoded_articles: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    pad_batch = DataCollatorWithPadding(tokenizer)

    return DataLoader(
        encoded_articles,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=pad_batch,
    )


def build_model(settings: dict) -> torch.nn.Module:
    number_of_labels = len(config.LABELS)

    if not settings["from_scratch"]:
        return AutoModelForSequenceClassification.from_pretrained(
            settings["checkpoint"], num_labels=number_of_labels
        )

    # Same architecture and tokenizer as the checkpoint, but much smaller
    # and with random weights, so it learns only from our articles
    small_config = AutoConfig.from_pretrained(
        settings["checkpoint"],
        num_labels=number_of_labels,
        num_hidden_layers=4,
        hidden_size=256,
        num_attention_heads=4,
        intermediate_size=1024,
    )

    return AutoModelForSequenceClassification.from_config(small_config)


def compute_class_weights(train: pd.DataFrame) -> torch.Tensor:
    train_ids = data_loading.labels_to_ids(train["category"])

    all_label_ids = np.arange(len(config.LABELS))

    weights = compute_class_weight("balanced", classes=all_label_ids, y=train_ids)

    return torch.tensor(weights, dtype=torch.float)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {name: values.to(device) for name, values in batch.items()}


def predict_logits(
    model: torch.nn.Module, data_loader: DataLoader, device: torch.device
) -> np.ndarray:
    use_mixed_precision = device.type == "cuda"

    model.eval()

    all_logits = []

    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)

            with torch.autocast(device_type=device.type, enabled=use_mixed_precision):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

            batch_logits = outputs.logits.float().cpu().numpy()

            all_logits.append(batch_logits)

    return np.concatenate(all_logits)


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    loss_function: torch.nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> float:
    use_mixed_precision = device.type == "cuda"

    model.train()

    total_loss = 0.0

    for batch in data_loader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=use_mixed_precision):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            loss = loss_function(outputs.logits.float(), batch["labels"])

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)

        scaler.update()

        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def copy_model_weights(model: torch.nn.Module) -> dict:
    return {
        name: values.detach().cpu().clone()
        for name, values in model.state_dict().items()
    }


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    validation_ids: np.ndarray,
    settings: dict,
    class_weights: torch.Tensor | None,
) -> dict[str, list[float]]:
    device = get_device()

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=0.01
    )

    total_steps = len(train_loader) * settings["epochs"]

    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if class_weights is not None:
        class_weights = class_weights.to(device)

    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    history = {"train_loss": [], "validation_loss": [], "validation_macro_f1": []}

    best_validation_loss = float("inf")

    best_weights = copy_model_weights(model)

    epochs_without_improvement = 0

    for epoch in range(1, settings["epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_function, scaler, device
        )

        validation_logits = predict_logits(model, validation_loader, device)

        validation_scores = compute_scores(validation_ids, softmax(validation_logits))

        history["train_loss"].append(train_loss)

        history["validation_loss"].append(validation_scores["log_loss"])

        history["validation_macro_f1"].append(validation_scores["macro_f1"])

        print(
            f"Epoch {epoch}: train loss {train_loss:.4f}, "
            f"validation loss {validation_scores['log_loss']:.4f}, "
            f"validation macro-F1 {validation_scores['macro_f1']:.4f}"
        )

        if validation_scores["log_loss"] < best_validation_loss:
            best_validation_loss = validation_scores["log_loss"]

            best_weights = copy_model_weights(model)

            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= settings["patience"]:
            print(f"Early stopping: no improvement for {settings['patience']} epochs")

            break

    model.load_state_dict(best_weights)

    return history


def run_experiment(
    settings: dict, train: pd.DataFrame, validation: pd.DataFrame
) -> tuple[torch.nn.Module, PreTrainedTokenizerBase, dict[str, list[float]]]:
    set_seed(settings["seed"])

    tokenizer = AutoTokenizer.from_pretrained(settings["checkpoint"])

    train_articles = encode_articles(tokenizer, train, settings["strategy"])

    validation_articles = encode_articles(tokenizer, validation, settings["strategy"])

    train_loader = build_data_loader(
        train_articles, tokenizer, settings["batch_size"], shuffle=True
    )

    validation_loader = build_data_loader(
        validation_articles, tokenizer, settings["batch_size"], shuffle=False
    )

    validation_ids = data_loading.labels_to_ids(validation["category"])

    class_weights = None

    if settings["class_weights"]:
        class_weights = compute_class_weights(train)

    model = build_model(settings)

    history = train_model(
        model, train_loader, validation_loader, validation_ids, settings, class_weights
    )

    return model, tokenizer, history


def predict_article_logits(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    data: pd.DataFrame,
    settings: dict,
) -> np.ndarray:
    articles = encode_articles(tokenizer, data, settings["strategy"])

    data_loader = build_data_loader(
        articles, tokenizer, settings["batch_size"], shuffle=False
    )

    return predict_logits(model, data_loader, get_device())


def clear_gpu_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
