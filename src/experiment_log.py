import json
from datetime import datetime

import pandas as pd

from src import config

# Each model gets its own log file, so teammates never edit the same file
# and git does not produce merge conflicts
EXPERIMENT_LOGS_DIRECTORY = config.ROOT / "results" / "experiment_logs"


def log_experiment(
    member: str,
    model_name: str,
    experiment_name: str,
    split_name: str,
    settings: dict,
    scores: dict[str, float],
    notes: str = "",
) -> None:
    row = {
        "notes": notes,
        "member": member,
        "split": split_name,
        "model": model_name,
        "experiment": experiment_name,
        "settings": json.dumps(settings),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    row.update(scores)

    new_row = pd.DataFrame([row])

    log_file = EXPERIMENT_LOGS_DIRECTORY / f"{model_name}.csv"

    if log_file.exists():
        model_log = pd.read_csv(log_file)

        model_log = pd.concat([model_log, new_row], ignore_index=True)
    else:
        model_log = new_row

    EXPERIMENT_LOGS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    model_log.to_csv(log_file, index=False)

    print(f"Logged experiment '{experiment_name}' to {log_file}")


def load_all_experiments() -> pd.DataFrame:
    log_files = sorted(EXPERIMENT_LOGS_DIRECTORY.glob("*.csv"))

    model_logs = [pd.read_csv(log_file) for log_file in log_files]

    return pd.concat(model_logs, ignore_index=True)
