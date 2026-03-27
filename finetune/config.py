import os


def _get_env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_env_range(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"{name} must contain exactly two comma-separated dates.")
    return parts


def _expand_path(value: str) -> str:
    return os.path.expanduser(value)

class Config:
    """
    Configuration class for the entire project.
    """

    def __init__(self):
        # =================================================================
        # Data & Feature Parameters
        # =================================================================
        # TODO: Update this path to your Qlib data directory.
        self.qlib_data_path = _expand_path(_get_env_str("KRONOS_QLIB_DATA_PATH", "~/.qlib/qlib_data/cn_data"))
        self.instrument = _get_env_str("KRONOS_INSTRUMENT", "csi300")

        # Overall time range for data loading from Qlib.
        self.dataset_begin_time = _get_env_str("KRONOS_DATASET_BEGIN_TIME", "2021-01-01")
        self.dataset_end_time = _get_env_str("KRONOS_DATASET_END_TIME", "2025-12-31")
        self.local_csv_dir = _expand_path(_get_env_str("KRONOS_LOCAL_CSV_DIR", "./zlab/data/daily"))

        # Sliding window parameters for creating samples.
        self.lookback_window = _get_env_int("KRONOS_LOOKBACK_WINDOW", 90)  # Number of past time steps for input.
        self.predict_window = _get_env_int("KRONOS_PREDICT_WINDOW", 10)  # Number of future time steps for prediction.
        self.max_context = _get_env_int("KRONOS_MAX_CONTEXT", 512)  # Maximum context length for the model.

        # Features to be used from the raw data.
        self.feature_list = ['open', 'high', 'low', 'close', 'vol', 'amt']
        # Time-based features to be generated.
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']

        # =================================================================
        # Dataset Splitting & Paths
        # =================================================================
        # These ranges use the strict full-horizon rule:
        # prediction_start_date must fall on or after the split start and
        # prediction_end_date must stay on or before the split end.
        # Preprocessing still adds backward context automatically.
        self.train_time_range = _get_env_range("KRONOS_TRAIN_TIME_RANGE", ["2021-01-01", "2024-06-30"])
        self.val_time_range = _get_env_range("KRONOS_VAL_TIME_RANGE", ["2024-07-01", "2024-12-31"])
        self.test_time_range = _get_env_range("KRONOS_TEST_TIME_RANGE", ["2025-01-01", "2025-12-31"])
        self.backtest_time_range = _get_env_range("KRONOS_BACKTEST_TIME_RANGE", ["2025-01-01", "2025-12-31"])

        # TODO: Directory to save the processed, pickled datasets.
        self.dataset_path = _expand_path(_get_env_str("KRONOS_DATASET_PATH", "./data/processed_datasets"))

        # =================================================================
        # Training Hyperparameters
        # =================================================================
        self.clip = _get_env_float("KRONOS_CLIP", 5.0)  # Clipping value for normalized data to prevent outliers.

        self.epochs = _get_env_int("KRONOS_EPOCHS", 30)
        self.log_interval = _get_env_int("KRONOS_LOG_INTERVAL", 100)  # Log training status every N batches.
        self.batch_size = _get_env_int("KRONOS_BATCH_SIZE", 50)  # Batch size per GPU.
        self.num_workers = _get_env_int("KRONOS_NUM_WORKERS", 2)

        # Number of samples to draw for one "epoch" of training/validation.
        # This is useful for large datasets where a true epoch is too long.
        self.n_train_iter = _get_env_int("KRONOS_N_TRAIN_ITER", 2000 * self.batch_size)
        self.n_val_iter = _get_env_int("KRONOS_N_VAL_ITER", 400 * self.batch_size)

        # Learning rates for different model components.
        self.tokenizer_learning_rate = _get_env_float("KRONOS_TOKENIZER_LR", 2e-4)
        self.predictor_learning_rate = _get_env_float("KRONOS_PREDICTOR_LR", 4e-5)
        self.future_only_loss = _get_env_bool("KRONOS_FUTURE_ONLY_LOSS", False)
        self.normalize_with_context_only = _get_env_bool("KRONOS_NORMALIZE_WITH_CONTEXT_ONLY", False)

        # Gradient accumulation to simulate a larger batch size.
        self.accumulation_steps = _get_env_int("KRONOS_ACCUMULATION_STEPS", 1)

        # AdamW optimizer parameters.
        self.adam_beta1 = _get_env_float("KRONOS_ADAM_BETA1", 0.9)
        self.adam_beta2 = _get_env_float("KRONOS_ADAM_BETA2", 0.95)
        self.adam_weight_decay = _get_env_float("KRONOS_ADAM_WEIGHT_DECAY", 0.1)

        # Miscellaneous
        self.seed = _get_env_int("KRONOS_SEED", 100)  # Global random seed for reproducibility.

        # =================================================================
        # Experiment Logging & Saving
        # =================================================================
        self.use_comet = _get_env_bool("KRONOS_USE_COMET", True) # Set to False if you don't want to use Comet ML
        self.use_tensorboard = _get_env_bool("KRONOS_USE_TENSORBOARD", True)
        self.tensorboard_subdir = _get_env_str("KRONOS_TENSORBOARD_SUBDIR", "tensorboard")
        self.tensorboard_flush_secs = _get_env_int("KRONOS_TENSORBOARD_FLUSH_SECS", 10)
        self.comet_config = {
            # It is highly recommended to load secrets from environment variables
            # for security purposes. Example: os.getenv("COMET_API_KEY")
            "api_key": _get_env_str("COMET_API_KEY", "YOUR_COMET_API_KEY"),
            "project_name": _get_env_str("COMET_PROJECT_NAME", "Kronos-Finetune-Demo"),
            "workspace": _get_env_str("COMET_WORKSPACE", "your_comet_workspace") # TODO: Change to your Comet ML workspace name
        }
        self.comet_tag = _get_env_str("KRONOS_COMET_TAG", "finetune_demo")
        self.comet_name = _get_env_str("KRONOS_COMET_NAME", "finetune_demo")

        # Base directory for saving model checkpoints and results.
        # Using a general 'outputs' directory is a common practice.
        self.save_path = _expand_path(_get_env_str("KRONOS_SAVE_PATH", "./outputs/models"))
        self.tokenizer_save_folder_name = _get_env_str("KRONOS_TOKENIZER_SAVE_FOLDER_NAME", "finetune_tokenizer_demo")
        self.predictor_save_folder_name = _get_env_str("KRONOS_PREDICTOR_SAVE_FOLDER_NAME", "finetune_predictor_demo")
        self.backtest_save_folder_name = _get_env_str("KRONOS_BACKTEST_SAVE_FOLDER_NAME", "finetune_backtest_demo")

        # Path for backtesting results.
        self.backtest_result_path = _expand_path(_get_env_str("KRONOS_BACKTEST_RESULT_PATH", "./outputs/backtest_results"))

        # =================================================================
        # Model & Checkpoint Paths
        # =================================================================
        # TODO: Update these paths to your pretrained model locations.
        # These can be local paths or Hugging Face Hub model identifiers.
        self.pretrained_tokenizer_path = _expand_path(_get_env_str("KRONOS_PRETRAINED_TOKENIZER_PATH", "path/to/your/Kronos-Tokenizer-base"))
        self.pretrained_predictor_path = _expand_path(_get_env_str("KRONOS_PRETRAINED_PREDICTOR_PATH", "path/to/your/Kronos-small"))

        # Paths to the fine-tuned models, derived from the save_path.
        # These will be generated automatically during training.
        self.finetuned_tokenizer_path = f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model"
        self.finetuned_predictor_path = f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"

        # A/B experiment convenience flags.
        self.skip_tokenizer_finetune = _get_env_bool("KRONOS_SKIP_TOKENIZER_FINETUNE", False)
        self.predictor_tokenizer_path = _get_env_str("KRONOS_PREDICTOR_TOKENIZER_PATH", "")
        self.freeze_predictor_for_ab = _get_env_bool("KRONOS_FREEZE_PREDICTOR_FOR_AB", False)
        self.predictor_train_last_ratio = _get_env_float("KRONOS_PREDICTOR_TRAIN_LAST_RATIO", 1.0)
        self.freeze_embedding = _get_env_bool("KRONOS_FREEZE_EMBEDDING", False)
        self.train_time_embedding = _get_env_bool("KRONOS_TRAIN_TIME_EMBEDDING", False)

        # =================================================================
        # Backtesting Parameters
        # =================================================================
        self.backtest_n_symbol_hold = _get_env_int("KRONOS_BACKTEST_N_SYMBOL_HOLD", 50)  # Number of symbols to hold in the portfolio.
        self.backtest_n_symbol_drop = _get_env_int("KRONOS_BACKTEST_N_SYMBOL_DROP", 5)  # Number of symbols to drop from the pool.
        self.backtest_hold_thresh = _get_env_int("KRONOS_BACKTEST_HOLD_THRESH", 5)  # Minimum holding period for a stock.
        self.inference_T = _get_env_float("KRONOS_INFERENCE_T", 0.6)
        self.inference_top_p = _get_env_float("KRONOS_INFERENCE_TOP_P", 0.9)
        self.inference_top_k = _get_env_int("KRONOS_INFERENCE_TOP_K", 0)
        self.inference_sample_count = _get_env_int("KRONOS_INFERENCE_SAMPLE_COUNT", 10)
        self.eval_batch_size = _get_env_int("KRONOS_EVAL_BATCH_SIZE", 128)
        self.backtest_batch_size = _get_env_int("KRONOS_BACKTEST_BATCH_SIZE", 1000)
        self.backtest_benchmark = self._set_benchmark(self.instrument)

    def _set_benchmark(self, instrument):
        dt_benchmark = {
            'csi800': "SH000906",
            'csi1000': "SH000852",
            'csi300': "SH000300",
        }
        if instrument in dt_benchmark:
            return dt_benchmark[instrument]
        else:
            raise ValueError(f"Benchmark not defined for instrument: {instrument}")
