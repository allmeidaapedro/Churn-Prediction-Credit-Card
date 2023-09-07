'''
This script aims to provide util functions to be used in components and pipeline.
'''

'''
Importing the libraries.
'''

# File handling.
import os
import pickle

# Debugging and verbose.
import sys
from src.exception import CustomException

# Data manipulation.
import pandas as pd
import numpy as np


def save_object(file_path, object):
    '''
    Save a Python object to a binary file using pickle serialization.

    This function takes an object and a file path as input and saves the object to the specified file using pickle
    serialization. If the directory of the file does not exist, it will be created.

    Args:
        file_path (str): The path to the file where the object will be saved.
        object_to_save: The Python object that needs to be saved.

    Raises:
        CustomException: If any exception occurs during the file saving process, a custom exception is raised with
                         the original exception details.

    Example:
        save_object("saved_object.pkl", my_data)

    Note:
        This function uses pickle to serialize the object. Be cautious when loading pickled data, as it can pose
        security risks if loading data from untrusted sources.
    '''

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_object:
            pickle.dump(object, file_object)
    
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    '''
    Load a Python object from a binary file using pickle deserialization.

    This function reads and deserializes a Python object from the specified binary file using pickle. It returns the
    loaded object.

    Args:
        file_path (str): The path to the file from which the object will be loaded.

    Returns:
        object: The Python object loaded from the file.

    Raises:
        CustomException: If any exception occurs during the file loading process, a custom exception is raised with
                         the original exception details.
    '''

    try:
        with open(file_path, 'rb') as file_object:
            return pickle.load(file_object)
        
    except Exception as e:
        raise CustomException(e, sys)
    