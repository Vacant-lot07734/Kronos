import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def plot_close_compare(dates, true_close, pred_close, title, save_path=None, show=True):
    """
    dates: 形如 numpy.datetime64 / pandas.Timestamp 列表
    true_close/pred_close: shape [7,]
    """
    plt.figure(figsize=(10, 4))
    plt.plot(dates, true_close, marker="o", label="True Close")
    plt.plot(dates, pred_close, marker="o", label="Pred Close")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=30)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)

    if show:
        plt.tight_layout()
        plt.show()
    else:
        plt.close()
