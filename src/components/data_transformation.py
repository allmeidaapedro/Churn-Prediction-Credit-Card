'''
This script aims to apply data cleaning and preprocessing to the data.
'''

# Debugging and verbose.
import sys
from src.exception import CustomException
from src.logger import logging

# File handling.
import os

# Data manipulation.
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Preprocessing.
from sklearn.pipeline import Pipeline
from src.modelling_utils import FeatureEngineer, OneHotFeatureEncoder, OrdinalFeatureEncoder, TargetFeatureEncoder, RecursiveFeatureEliminator, ColumnDropper
from lightgbm import LGBMClassifier

# Utils.
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    '''
    Configuration class for data transformation.

    This data class holds configuration parameters related to data transformation. It includes attributes such as
    `preprocessor_file_path` that specifies the default path to save the preprocessor object file.

    Attributes:
        preprocessor_file_path (str): The default file path for saving the preprocessor object. By default, it is set
                                     to the 'artifacts' directory with the filename 'preprocessor.pkl'.

    Example:
        config = DataTransformationConfig()
        print(config.preprocessor_file_path)  # Output: 'artifacts/preprocessor.pkl'

    Note:
        This class uses the @dataclass decorator to automatically generate special methods like __init__ and __repr__
        based on the defined attributes.
    '''

    preprocessor_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    '''
    Data transformation class for preprocessing and transformation of train and test sets.

    This class handles the preprocessing and cleaning of datasets, including
    feature engineering and scaling.

    :ivar data_transformation_config: Configuration instance for data transformation.
    :type data_transformation_config: DataTransformationConfig
    '''
    def __init__(self) -> None:
        '''
        Initialize the DataTransformation instance with a DataTransformationConfig.
        '''
        self.data_transformation_config = DataTransformationConfig()


    def get_preprocessor(self):
        '''
        Get a preprocessor for data transformation.

        This method sets up pipelines for ordinal encoding, target encoding,
        and scaling of features.

        :return: Preprocessor object for data transformation.
        :rtype: ColumnTransformer
        :raises CustomException: If an exception occurs during the preprocessing setup.
        '''

        try:
            # Construct the preprocessor for tree-based models.
            one_hot_encoding_features = ['gender']

            # I will encode 'unknown' as the last one, due to its churn rate (among the first or second highest one).
            ordinal_encoding_orders = {
                'education_level': ['Uneducated',
                                    'High School',
                                    'College',
                                    'Graduate',
                                    'Post-Graduate',
                                    'Doctorate',
                                    'Unknown'],
                'income_category': ['Less than $40K',
                                    '$40K - $60K',
                                    '$60K - $80K',
                                    '$80K - $120K',
                                    '$120K +',
                                    'Unknown'],
                'card_category': ['Blue',
                                'Silver',
                                'Gold',
                                'Platinum']
            }

            target_encoding_features = ['marital_status']

            to_drop_features = ['naive_bayes_classifier_attrition_flag_card_category_contacts_count_12_mon_dependent_count_education_level_months_inactive_12_mon_1', 
                                'naive_bayes_classifier_attrition_flag_card_category_contacts_count_12_mon_dependent_count_education_level_months_inactive_12_mon_2',
                                'clientnum',
                                'avg_open_to_buy']

            logging.info(f'Categorical features: {one_hot_encoding_features+list(ordinal_encoding_orders.keys()) + target_encoding_features}.')
            logging.info(f'One-hot encoding applied to {one_hot_encoding_features}.')
            logging.info(f'Ordinal encoding applied to: {list(ordinal_encoding_orders.keys())}.')
            logging.info(f'Target encoding applied to: {target_encoding_features}.')

            preprocessor = Pipeline(
                steps=[
                    ('feature_engineer', FeatureEngineer()),
                    ('one_hot_encoder', OneHotFeatureEncoder(to_encode=one_hot_encoding_features)),
                    ('ordinal_encoder', OrdinalFeatureEncoder(to_encode=ordinal_encoding_orders)),
                    ('target_encoder', TargetFeatureEncoder(to_encode=target_encoding_features)),
                    ('col_dropper', ColumnDropper(to_drop=to_drop_features)),
                    ('rfe_selector', RecursiveFeatureEliminator(n_folds=5, 
                                                                scoring='roc_auc',
                                                                estimator=LGBMClassifier()))
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
    
    
    def apply_data_transformation(self, train_path, test_path):
        '''
        Apply data transformation process.

        Reads, preprocesses, and transforms training and test datasets.

        :param train_path: Path to the training dataset CSV file.
        :param test_path: Path to the test dataset CSV file.
        :return: Prepared training and test datasets and the preprocessor file path.
        :rtype: tuple
        :raises CustomException: If an exception occurs during the data transformation process.
        '''
        
        try:

            logging.info('Read training and test sets.')

            # Obtain train and test entire sets from artifacts.
            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)

            logging.info('Obtain preprocessor object.')

            preprocessor = self.get_preprocessor()

            # Get train and test predictor and target sets.
            X_train = train.drop(columns=['churn_flag'])
            y_train = train['churn_flag'].copy()

            X_test = test.drop(columns=['churn_flag'])
            y_test = test['churn_flag'].copy()

            logging.info('Preprocess training and test sets.')

            X_train_prepared = preprocessor.fit_transform(X_train, y_train)
            X_test_prepared = preprocessor.transform(X_test)
            
            print(f'Final columns: {X_train_prepared.columns.tolist()}')

            # Get final training and test entire prepared sets.
            train_prepared = pd.concat([X_train_prepared, y_train.reset_index(drop=True)], axis=1)
            test_prepared = pd.concat([X_test_prepared, y_test.reset_index(drop=True)], axis=1)

            logging.info('Save preprocessing object.')

            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path,
                object=preprocessor
            )
        
            return train_prepared, test_prepared, self.data_transformation_config.preprocessor_file_path
        
        except Exception as e:
            raise CustomException(e, sys)
        