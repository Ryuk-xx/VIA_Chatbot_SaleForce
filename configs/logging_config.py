import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from datetime import datetime
from configs.config import load_config
APP_CONFIG = load_config()

PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs"

def setup_logging(logger_name = "Chatbot_SaleForce"):
    """Configures logging based on settings in config.yaml."""
    log_config = APP_CONFIG.get('logging', {})
    log_level_str = log_config.get('level', 'INFO').upper()

    # Ensure log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Tạo tên file log theo ngày
    log_filename = f"Chatbot_SaleForce_{datetime.now().strftime('%Y-%m-%d')}.log"
    log_file_path = LOG_DIR / log_filename

    log_level = getattr(logging, log_level_str, logging.INFO)
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    formatter = logging.Formatter(log_format)

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    file_handler = TimedRotatingFileHandler(
        filename=log_file_path,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding='utf-8',
        utc=False  # đổi thành True nếu server bạn dùng UTC
    )
    file_handler.suffix = "%Y-%m-%d.log"  # suffix cho các file backup
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info("Logging setup complete.")
    return logger
