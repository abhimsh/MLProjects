import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_file_path = os.path.join(os.getcwd(), "logs")

os.makedirs(log_file_path, exist_ok=True)

logging.basicConfig(filename=os.path.join(log_file_path, LOG_FILE),
                    filemode="w",
                    format="%(asctime)s [%(filename)s:%(lineno)d]  %(name)s - %(levelname)s - %(message)s",
                    level=logging.DEBUG)