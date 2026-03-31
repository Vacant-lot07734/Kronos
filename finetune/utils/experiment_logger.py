import json
import os


def _to_scalar(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return float(value)


class ExperimentLogger:
    def __init__(self, comet_logger=None, tensorboard_writer=None, tensorboard_log_dir: str | None = None):
        self.comet_logger = comet_logger
        self.tensorboard_writer = tensorboard_writer
        self.tensorboard_log_dir = tensorboard_log_dir

    def __bool__(self):
        return self.comet_logger is not None or self.tensorboard_writer is not None

    @property
    def has_comet(self) -> bool:
        return self.comet_logger is not None

    @property
    def has_tensorboard(self) -> bool:
        return self.tensorboard_writer is not None

    def log_parameters(self, config: dict):
        if self.comet_logger is not None:
            self.comet_logger.log_parameters(config)
        if self.tensorboard_writer is not None:
            config_text = json.dumps(config, indent=2, ensure_ascii=False, default=str)
            self.tensorboard_writer.add_text("run/config", f"```json\n{config_text}\n```", global_step=0)

    def log_text(self, tag: str, text: str, step: int = 0):
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_text(tag, text, global_step=step)

    def log_metric(self, name: str, value, step: int | None = None, epoch: int | None = None, tb_tag: str | None = None):
        scalar = _to_scalar(value)
        if self.comet_logger is not None:
            kwargs = {}
            if step is not None:
                kwargs["step"] = step
            if epoch is not None:
                kwargs["epoch"] = epoch
            self.comet_logger.log_metric(name, scalar, **kwargs)

        if self.tensorboard_writer is not None:
            global_step = step if step is not None else epoch
            self.tensorboard_writer.add_scalar(tb_tag or name, scalar, global_step=global_step)

    def log_model(self, name: str, path: str):
        if self.comet_logger is not None and hasattr(self.comet_logger, "log_model"):
            self.comet_logger.log_model(name, path)
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_text(f"artifacts/{name}", path, global_step=0)

    def close(self):
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.flush()
            self.tensorboard_writer.close()
        if self.comet_logger is not None:
            self.comet_logger.end()


def _create_comet_logger(config: dict):
    if not config.get("use_comet"):
        return None

    try:
        from comet_ml import Experiment
    except ImportError as exc:
        raise ImportError(
            "comet_ml is not installed, but use_comet=True. "
            "Install comet_ml or disable Comet logging."
        ) from exc

    logger = Experiment(
        api_key=config["comet_config"]["api_key"],
        project_name=config["comet_config"]["project_name"],
        workspace=config["comet_config"]["workspace"],
    )
    logger.add_tag(config["comet_tag"])
    logger.set_name(config["comet_name"])
    return logger


def _create_tensorboard_writer(config: dict, save_dir: str):
    if not config.get("use_tensorboard", True):
        return None, None

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print(
            "WARN: tensorboard is not installed. TensorBoard logging is disabled for this run. "
            "Install it with `python -m pip install tensorboard` if you want local dashboards."
        )
        return None, None

    log_dir = os.path.join(save_dir, config.get("tensorboard_subdir", "tensorboard"))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(
        log_dir=log_dir,
        flush_secs=int(config.get("tensorboard_flush_secs", 10)),
    )
    return writer, log_dir


def build_experiment_logger(config: dict, save_dir: str):
    comet_logger = _create_comet_logger(config)
    tensorboard_writer, tensorboard_log_dir = _create_tensorboard_writer(config, save_dir)
    logger = ExperimentLogger(
        comet_logger=comet_logger,
        tensorboard_writer=tensorboard_writer,
        tensorboard_log_dir=tensorboard_log_dir,
    )
    logger.log_parameters(config)
    return logger
