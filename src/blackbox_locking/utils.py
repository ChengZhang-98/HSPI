from pathlib import Path
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


ROOT_DIR = Path(__file__).resolve().parents[0]  # path/to/src/blackbox_locking
