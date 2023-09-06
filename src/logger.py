'''
This script aims to define a logger with a standard logging message configuration.
Logging messages are employed to track application behavior, errors, and events. They facilitate debugging, provide insight into program flow, and aid in monitoring and diagnosing issues. Logging enhances code quality, troubleshooting, and maintenance.
'''

'''
Importing the libraries.
'''

# Debugging and verbose.
import logging
from datetime import datetime

# File handling.
import os


## The filename for the log file, including the current date and time.
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

## The path to the directory where log files are stored.
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)

## Create the directory if it doesn't exist.
os.makedirs(logs_path,exist_ok=True)


## The full path to the log file.
LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)


## Configure the logging settings.
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)