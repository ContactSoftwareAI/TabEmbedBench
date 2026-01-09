import logging
from datetime import datetime
from pathlib import Path

from google.cloud import storage

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
    capture_warnings=True,
):
    """
    Sets up unified logging for the application.

    This function configures logging with both console and optional file handlers.
    It applies a specific logging format, sets logging levels for various loggers,
    and optionally captures warnings as part of the logging process.

    Args:
        save_logs: Determines whether logs should be saved to a file.
        log_dir: Specifies the directory where log files will be stored.
        timestamp: Provides a timestamp to include in the log filename for uniqueness.
        logging_level: Indicates the logging level to be applied to all configured loggers.
        capture_warnings: Specifies whether to capture Python warnings as part of logging.

    Returns:
        str: The file path of the log file if save_logs is True, otherwise None.
    """
    handlers = [logging.StreamHandler()]
    log_file = None

    if save_logs:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

        log_file = log_path / f"benchmark_complete_{timestamp}.log"
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%("
            "lineno)d - %(message)s"
        ),
        handlers=handlers,
        force=True,
    )

    if capture_warnings:
        logging.captureWarnings(True)
        warnings_logger = logging.getLogger("py.warnings")
        warnings_logger.setLevel(logging_level)

    outlier_logger = logging.getLogger("TabEmbedBench_Outlier")
    tabarena_logger = logging.getLogger("TabEmbedBench_TabArena")
    dataset_separation = logging.getLogger("TabEmbedBench_DatasetSeparation")
    main_logger = logging.getLogger("TabEmbedBench_Main")

    for logger in [outlier_logger, tabarena_logger, dataset_separation, main_logger]:
        logger.setLevel(logging_level)

    return log_file


def upload_logs_to_gcs(local_log_file: Path, bucket_name: str, gcs_path: str):
    """Uploads the local log file to Google Cloud Storage."""
    if not local_log_file or not local_log_file.exists():
        return

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        # Construct the full path: e.g., ijcai/logs/benchmark_complete_timestamp.log
        blob_path = f"{gcs_path}/logs/{local_log_file.name}"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(str(local_log_file))
        print(f"Successfully uploaded logs to gs://{bucket_name}/{blob_path}")
    except Exception as e:
        logging.getLogger("TabEmbedBench_Main").error(
            f"Failed to upload logs to GCS: {e}"
        )


def get_benchmark_logger(name: str) -> logging.Logger:
    """Get a logger with custom benchmark logging methods."""
    return logging.getLogger(name)
