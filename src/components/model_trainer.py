'''
This script aims to train and save the selected final model from the modelling notebook.
'''

'''
Importing the libraries
'''

# File handling.
import os
from dataclasses import dataclass

# Debugging and verbose.
import sys
from src.exception import CustomException
from src.logger import logging

# Modelling.
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Utils.
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    '''
    Configuration class for model training.

    This data class holds configuration parameters related to model training. It includes attributes such as
    `model_file_path` that specifies the default path to save the trained model file.

    Attributes:
        model_file_path (str): The default file path for saving the trained model. By default, it is set to the
                              'artifacts' directory with the filename 'model.pkl'.

    Example:
        config = ModelTrainerConfig()
        print(config.model_file_path)  # Output: 'artifacts/model.pkl'

    Note:
        This class uses the @dataclass decorator to automatically generate special methods like __init__ and __repr__
        based on the defined attributes.
    '''

    model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    '''
    This class is responsible for training and saving the best XGBoost model
    using the best hyperparameters found in modelling notebook analysis with stratified 
    k-fold cross-validation and Bayesian optimization.

    Attributes:
        model_trainer_config (ModelTrainerConfig): An instance of `ModelTrainerConfig` for configuration settings.

    Methods:
        apply_model_trainer(train_prepared, test_prepared):
            Trains the best XGBoost model using the provided prepared training and testing data,
            and returns the classification report and ROC-AUC score on the test set.

    '''

    def __init__(self) -> None:
        '''
        Initializes a new instance of the `ModelTrainer` class.

        Attributes:
            model_trainer_config (ModelTrainerConfig): An instance of `ModelTrainerConfig` for configuration settings.
        '''
        self.model_trainer_config = ModelTrainerConfig()
    
    
    def apply_model_trainer(self, train_prepared, test_prepared):
        '''
        Trains the best XGBoost model using the provided prepared training and testing data, 
        the best hyperparameters found during the modelling notebook analysis using stratified k-fold
        cross validation and bayesian optimization and returns the classification report and ROC-AUC 
        score on the test set.

        Args:
            train_prepared (numpy.ndarray): The prepared training data.
            test_prepared (numpy.ndarray): The prepared testing data.

        Returns:
            str: The classification report of the best model on the test set.
            float: The ROC-AUC score of the best model on the test set.

        Raises:
            CustomException: If an error occurs during the training and evaluation process.

        '''

        try:
            logging.info('Split train and test prepared arrays.')
            
            X_train_prepared, X_test_prepared, y_train, y_test = train_prepared[:, :-1], test_prepared[:, :-1], train_prepared[:, -1], test_prepared[:, -1]

            logging.info('Started to train the best XGBoost model with the best hyparameters found in modelling step.')

            best_params = {
                'colsample_bytree': 0.6990559033956983,
                'gamma': 9.029710156344564,
                'lambda': 2.3076689805633372,
                'learning_rate': 0.12498672387208519,
                'max_depth': 4,
                'min_child_weight': 3,
                'n_estimators': 500,
                'scale_pos_weight': 10,
                'subsample': 0.7541369107532966
            }

            best_model = XGBClassifier(**best_params)

            best_model.fit(X_train_prepared, y_train)

            logging.info('Saving the best model.')

            save_object(
                file_path=self.model_trainer_config.model_file_path,
                object=best_model
            )

            logging.info('Best model classification report and roc-auc score on test set returned.')

            final_predictions = best_model.predict(X_test_prepared)
            
            class_report = classification_report(y_test, final_predictions)
            auc_score = roc_auc_score(y_test, final_predictions)

            return class_report, auc_score

        except Exception as e:
            raise CustomException(e, sys)