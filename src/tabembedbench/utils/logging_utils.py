import logging
from datetime import datetime
from pathlib import Path

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

NEIGHBORS_PROGRESS = 5

logging.addLevelName(NEIGHBORS_PROGRESS, "NEIGHBORS_PROGRESS")

def neighbors_progress(self, message, *args, **kwargs):
    """Log 'message % args' with severity 'NEIGHBORS_PROGRESS'."""
    if self.isEnabledFor(NEIGHBORS_PROGRESS):
        self._log(NEIGHBORS_PROGRESS, message, args, **kwargs)

logging.Logger.neighbors_progress = neighbors_progress

def setup_unified_logging(
    save_logs: bool = True,
    log_dir="log",
    timestamp=timestamp,
    logging_level=logging.INFO,
):
    handlers = [logging.StreamHandler()]
    log_file = None

    if save_logs:
        # Create log directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

        # Create comprehensive log file
        log_file = log_path / f"benchmark_complete_{timestamp}.log"
        handlers.append(logging.FileHandler(log_file))

    # Configure root logger to capture all loggers
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        handlers=handlers,
        force=True,  # Override any existing configuration
    )

    # Get references to all your benchmark loggers
    outlier_logger = logging.getLogger("TabEmbedBench_Outlier")
    tabarena_logger = logging.getLogger("TabEmbedBench_TabArena")
    main_logger = logging.getLogger("TabEmbedBench_Main")

    # Ensure they all use the same handlers and level
    for logger in [outlier_logger, tabarena_logger, main_logger]:
        logger.setLevel(logging_level)

    return log_file

def get_benchmark_logger(name: str) -> logging.Logger:
    """Get a logger with custom benchmark logging methods."""
    return logging.getLogger(name)

