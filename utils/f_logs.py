import os, logging
from logging.handlers import TimedRotatingFileHandler

def get_train_logger(log_path="./logs/dqn-train.log", fmt_extra=""):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("dqn.train")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fh = TimedRotatingFileHandler(log_path, when="midnight", backupCount=7, encoding="utf-8", utc=True)
    fmt = logging.Formatter(
        fmt=f"%(asctime)s %(levelname)s [%(name)s] {fmt_extra}%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    return logger
