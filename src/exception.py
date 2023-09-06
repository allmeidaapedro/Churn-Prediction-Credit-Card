'''
This script aims to define custom exception for exception handling purposes. It will be easier to identify errors and issues.
'''

'''
Importing the libraries.
'''

# Debugging and verbose.
import sys
from src.logger import logging


def detailed_error_msg(error, error_details: sys):
    '''
    Generate a detailed error message including file name, line number, and error message.
    
    Args:
        error: The original error or exception.
        error_details (sys): System information about the error.
        
    Returns:
        str: A detailed error message.
    '''
    _, _, exception_traceback = error_details.exc_info()
    file_name = exception_traceback.tb_frame.f_code.co_filename
    line_number = exception_traceback.tb_lineno
    detailed_error_message = f'An error occurred in python file [{file_name}] line number [{line_number}] error message [{str(error)}]'
    
    return detailed_error_message


class CustomException(Exception):
    '''
    Custom exception class with detailed error information.
    '''
    def __init__(self, detailed_error_message: str, error_details: sys) -> None:
        '''
        Initialize a DetailedException instance.

        Args:
            detailed_error_message (str): The detailed error message.
            error_details (sys): System information about the error.
        '''
        super().__init__(detailed_error_message)
        self.detailed_error_message = detailed_error_msg(detailed_error_message, error_details=error_details)


    def __str__(self) -> str:
        '''
        Convert the exception to a string representation.
        
        Returns:
            str: The detailed error message.
        '''
        return self.detailed_error_message
    