'''
This script aims to read the dataset from the data source and split it into train and test sets. These steps results are necessary for data transformation component.
'''

'''
Importing libraries
'''


# Debugging, verbose.
import sys
from src.exception import CustomException
from src.logger import logging

# File handling.
import os

# Data manipulation.
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Warnings.
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class DataIngestionConfig:
    '''
    Configuration class for data ingestion.

    This data class holds configuration parameters related to data ingestion. It includes attributes such as
    `train_data_path`, `test_data_path`, and `raw_data_path` that specify the default paths for train, test, and raw
    data files respectively.

    Attributes:
        train_data_path (str): The default file path for the training data. By default, it is set to the 'artifacts'
                              directory with the filename 'train.csv'.
        test_data_path (str): The default file path for the test data. By default, it is set to the 'artifacts'
                             directory with the filename 'test.csv'.
        raw_data_path (str): The default file path for the raw data. By default, it is set to the 'artifacts'
                            directory with the filename 'data.csv'.

    Example:
        config = DataIngestionConfig()
        print(config.train_data_path)  # Output: 'artifacts/train.csv'

    Note:
        This class uses the @dataclass decorator to automatically generate special methods like __init__ and __repr__
        based on the defined attributes.
    '''

    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    raw_data_path = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    '''
    Data ingestion class for preparing and splitting datasets.

    This class handles the data ingestion process, including reading, splitting,
    and saving the raw data, training data, and test data.

    :ivar ingestion_config: Configuration instance for data ingestion.
    :type ingestion_config: DataIngestionConfig
    '''

    def __init__(self) -> None:
        '''
        Initialize the DataIngestion instance with a DataIngestionConfig.
        '''
        self.ingestion_config = DataIngestionConfig()

    def apply_data_ingestion(self):
        '''
        Apply data ingestion process.

        Reads the dataset, performs train-test split while stratifying the target,
        and saves raw data, training data, and test data.

        :return: Paths to the saved training data and test data files.
        :rtype: tuple
        :raises CustomException: If an exception occurs during the data ingestion process.
        '''
        
        try:
            logging.info('Read the dataset as a Pandas DataFrame and save it as a csv.')

            data_path = '../input/BankChurners.csv'
            df = pd.read_csv(data_path)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Obtain X and y.')

            df.columns = [x.lower() for x in df.columns]
            df.rename(columns={'attrition_flag': 'churn_flag'}, inplace=True)
            df['churn_flag'] = df['churn_flag'].map({'Attrited Customer': 1, 'Existing Customer': 0})

            X = df.drop(columns=['churn_flag'])
            y = df['churn_flag'].copy()

            logging.info('Split the data intro training and test sets.')

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            # Get back train and test entire sets.
            train = pd.concat([X_train, y_train], axis=1)
            test = pd.concat([X_test, y_test], axis=1)

            logging.info('Save training and test sets into a csv.')
            
            train.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        
        except Exception as e:
            raise CustomException(e, sys)
        