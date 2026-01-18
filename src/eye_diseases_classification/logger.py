from loguru import logger
import sys
from pathlib import Path

# Remove default logger
logger.remove()

# Terminal: only warnings and above
logger.add(sys.stdout, level="WARNING")

# File logging: debug and above, rotated at 100 MB
Path("logs").mkdir(exist_ok=True)
logger.add("logs/my_log.log", level="DEBUG", rotation="100 MB")
