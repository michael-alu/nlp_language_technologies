import random
import sys

import numpy as np

FINAL_RUN_SEEDS = [42, 7, 2026]


def set_seed(seed: int) -> None:
    random.seed(seed)

    np.random.seed(seed)

    # Only seed the deep learning library your notebook has already imported
    if "torch" in sys.modules:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if "tensorflow" in sys.modules:
        import tensorflow as tf

        tf.random.set_seed(seed)

    print(f"Random seed set to {seed}")
