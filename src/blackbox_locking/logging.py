import logging

from colorlog import ColoredFormatter

formatter = ColoredFormatter(
    "%(log_color)s%(levelname)-8s[%(filename)s:%(lineno)d | %(asctime)s]%(reset)s %(blue)s%(message)s",
    datefmt="%Y-%m-%d_%H:%M:%S",
    reset=True,
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
    secondary_log_colors={},
    style="%",
)

handler = logging.StreamHandler()
handler.setFormatter(formatter)

root_logger = logging.getLogger("blackbox_locking")
root_logger.addHandler(handler)
root_logger.setLevel(logging.INFO)
root_logger.propagate = False


def set_logging_verbosity(level: str = logging.INFO):
    match level:
        case logging.DEBUG:
            root_logger.setLevel(logging.DEBUG)
        case logging.INFO:
            root_logger.setLevel(logging.INFO)
        case logging.WARNING:
            root_logger.setLevel(logging.WARNING)
        case logging.ERROR:
            root_logger.setLevel(logging.ERROR)
        case logging.CRITICAL:
            root_logger.setLevel(logging.CRITICAL)
        case _:
            raise ValueError(
                f"Unknown logging level: {level}, should be one of: logging.DEBUG/INFO/WARNING/ERROR/CRITICAL"
            )
    root_logger.info(f"Set logging level to {level}")


def get_logger(name: str):
    return root_logger.getChild(name)
