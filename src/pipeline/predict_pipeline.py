'''
This script aims to create the predict pipeline for a simple web application which will be interacting with the pkl files, such that we can make predictions by giving values of input features. 
'''

# Debugging and verbose.
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object

# Data manipulation.
import pandas as pd

# File handling.
import os


class PredictPipeline:
    '''
    Class for making predictions using a trained model and preprocessor.

    This class provides a pipeline for making predictions on new instances using a trained machine learning model and
    a preprocessor. It loads the model and preprocessor from files, preprocesses the input features, and makes predictions.

    Methods:
        predict(features):
            Make predictions on new instances using the loaded model and preprocessor.

    Example:
        pipeline = PredictPipeline()
        new_features = [...]
        prediction = pipeline.predict(new_features)

    Note:
        This class assumes the availability of the load_object function.
    '''

    
    def __init__(self) -> None:
        '''
        Initializes a PredictPipeline instance.

        Initializes the instance. No specific setup is required in the constructor.
        '''
        pass


    def predict(self, features):
        '''
        Make predictions on new instances using the loaded model and preprocessor.

        Args:
            features: Input features for which predictions will be made.

        Returns:
            predictions: Predicted labels for the input features.

        Raises:
            CustomException: If an exception occurs during the prediction process.
        '''
        try:


            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            logging.info('Load model and preprocessor objects.')

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            logging.info('Preprocess the input data.')

            prepared_data = preprocessor.transform(features)
            
            # Assert input data is float64 data type.
            prepared_data = prepared_data.astype('float64')
            
            logging.info('Predict.')
            
            # Predict customer's churn probability.
            predicted_proba = model.predict_proba(prepared_data)[:, 1][0]

            # Prediction output (customer's probability of churning).
            prediction = f"""Customer's probability of churning:
                             {round(predicted_proba * 100, 3)}%"""

            logging.info('Prediction successfully made.')

            return prediction

        except Exception as e:
            raise CustomException(e, sys)
        

class InputData:
    '''
    Class for handling input data for predictions.

    This class provides a structured representation for input data that is meant to be used for making predictions.
    It maps input variables from HTML inputs to class attributes and provides a method to convert the input data into
    a DataFrame format suitable for making predictions.

    Attributes:
        customer_age (int)              
        gender (str)                      
        dependent_count (int)             
        education_level (str)             
        marital_status (str)              
        income_category (str)             
        card_category (str)               
        months_on_book (int)              
        total_relationship_count (int)    
        months_inactive_12_mon (int)      
        contacts_count_12_mon (int)       
        credit_limit (float)                
        total_revolving_bal (int)         
        total_amt_chng_q4_q1 (float)       
        total_trans_amt (int)            
        total_trans_ct (int)              
        total_ct_chng_q4_q1 (float)         
        avg_utilization_ratio (float)       

    Methods:
        get_input_data_df():
            Convert the mapped input data into a DataFrame for predictions.

    Note:
        This class assumes the availability of the pandas library and defines the CustomException class.
    '''

    def __init__(self,
                 customer_age: int,
                 gender: str,
                 dependent_count: int,
                 education_level: str,
                 marital_status: str,
                 income_category: str,
                 card_category: str,
                 months_on_book: int,
                 total_relationship_count: int,
                 months_inactive_12_mon: int,
                 contacts_count_12_mon: int,
                 credit_limit: float,
                 total_revolving_bal: int,
                 total_amt_chng_q4_q1: float,
                 total_trans_amt: int,
                 total_trans_ct: int,
                 total_ct_chng_q4_q1: float,
                 avg_utilization_ratio: float) -> None:
        '''
        Initialize an InputData instance with mapped input data.

        Args:
            customer_age (int)              
            gender (str)                      
            dependent_count (int)             
            education_level (str)             
            marital_status (str)              
            income_category (str)             
            card_category (str)               
            months_on_book (int)              
            total_relationship_count (int)    
            months_inactive_12_mon (int)      
            contacts_count_12_mon (int)       
            credit_limit (float)                
            total_revolving_bal (int)         
            total_amt_chng_q4_q1 (float)       
            total_trans_amt (int)            
            total_trans_ct (int)              
            total_ct_chng_q4_q1 (float)         
            avg_utilization_ratio (float)
        '''
        
        # Map variables from html inputs.
        self.customer_age = customer_age
        self.gender = gender
        self.dependent_count = dependent_count
        self.education_level = education_level
        self.marital_status = marital_status
        self.income_category = income_category
        self.card_category = card_category
        self.months_on_book = months_on_book
        self.total_relationship_count = total_relationship_count
        self.months_inactive_12_mon = months_inactive_12_mon
        self.contacts_count_12_mon = contacts_count_12_mon
        self.credit_limit = credit_limit
        self.total_revolving_bal = total_revolving_bal
        self.total_amt_chng_q4_q1 = total_amt_chng_q4_q1
        self.total_trans_amt = total_trans_amt
        self.total_trans_ct = total_trans_ct
        self.total_ct_chng_q4_q1 = total_ct_chng_q4_q1
        self.avg_utilization_ratio = avg_utilization_ratio


    def get_input_data_df(self):
        '''
        Convert the mapped input data into a DataFrame for predictions.

        Returns:
            input_data_df (DataFrame): DataFrame containing the mapped input data.

        Raises:
            CustomException: If an exception occurs during the process.
        '''
        try:
            input_data_dict = dict()

            # Map the variables to the form of a dataframe for being used in predictions.
            
            input_data_dict['customer_age'] = [self.customer_age]
            input_data_dict['gender'] = [self.gender]
            input_data_dict['dependent_count'] = [self.dependent_count]
            input_data_dict['education_level'] = [self.education_level]
            input_data_dict['marital_status'] = [self.marital_status]
            input_data_dict['income_category'] = [self.income_category]
            input_data_dict['card_category'] = [self.card_category]
            input_data_dict['months_on_book'] = [self.months_on_book]
            input_data_dict['total_relationship_count'] = [self.total_relationship_count]
            input_data_dict['months_inactive_12_mon'] = [self.months_inactive_12_mon]
            input_data_dict['contacts_count_12_mon'] = [self.contacts_count_12_mon]
            input_data_dict['credit_limit'] = [self.credit_limit]
            input_data_dict['total_revolving_bal'] = [self.total_revolving_bal]
            input_data_dict['total_amt_chng_q4_q1'] = [self.total_amt_chng_q4_q1]
            input_data_dict['total_trans_amt'] = [self.total_trans_amt]
            input_data_dict['total_trans_ct'] = [self.total_trans_ct]
            input_data_dict['total_ct_chng_q4_q1'] = [self.total_ct_chng_q4_q1]
            input_data_dict['avg_utilization_ratio'] = [self.avg_utilization_ratio]

            input_data_df = pd.DataFrame(input_data_dict)

            return input_data_df
        
        except Exception as e:
            raise CustomException(e, sys)