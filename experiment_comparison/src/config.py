from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import yaml

@dataclass
class Config:
    csv_dir: str
    pattern: str
    max_files: Optional[int]

    date_col: str
    features: List[str]

    lookback: int
    pred_len: int

    train_end: str
    val_end: str

    batch_size: int
    epochs: int
    lr: float
    patience: int
    num_workers: int

    hidden_dim: int
    num_layers: int
    dropout: float

    ckpt_dir: str
    ckpt_name: str

    # ✅ 新增：Informer 的配置（允许是 dict）
    informer: Optional[Dict[str, Any]] = None

    @staticmethod
    def from_yaml(path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)

        # 兼容：如果 yaml 没有 informer，就给 None
        if "informer" not in d:
            d["informer"] = None

        return Config(**d)
