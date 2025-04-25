import logging
import os
import datetime



class Logger:
    def __init__(self, log_path):
        self.error_log = os.path.join(log_path, "ErrorLog.log")

        self.logger = logging.getlogger("ErrorLogger")
        self.logger.setLevel(logging.ERROR)
        if not self.logger.handlers:
            file_handler = logging.FileHandler(self.error_log)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def log_error(self, message):
        self.logger.error(message)

    def log_info(self, message):
        self.logger.info(message)
    def log_warning(self, message):
        self.logger.warning(message)



