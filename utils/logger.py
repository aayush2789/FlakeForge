import logging
import sys
from pathlib import Path
from typing import Optional

# ANSI Color Codes
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class ColorFormatter(logging.Formatter):
    """Custom Formatter to add colors to log levels."""
    
    FORMATS = {
        logging.DEBUG: Colors.OKBLUE + "%(levelname)s: %(message)s" + Colors.ENDC,
        logging.INFO: Colors.OKGREEN + "%(levelname)s: %(message)s" + Colors.ENDC,
        logging.WARNING: Colors.WARNING + "%(levelname)s: %(message)s" + Colors.ENDC,
        logging.ERROR: Colors.FAIL + "%(levelname)s: %(message)s" + Colors.ENDC,
        logging.CRITICAL: Colors.FAIL + Colors.BOLD + "%(levelname)s: %(message)s" + Colors.ENDC
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def get_logger(name: str, level: int = logging.INFO, log_file: Optional[Path] = None) -> logging.Logger:
    """Returns a color-coded logger instance, optionally logging to a file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColorFormatter())
        logger.addHandler(console_handler)
        
        # File handler without colors (cleaner for logs)
        if log_file:
            from pathlib import Path
            file_handler = logging.FileHandler(str(log_file), encoding='utf-8')
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
    
    return logger
