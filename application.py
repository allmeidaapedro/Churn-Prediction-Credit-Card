'''
This script aims to build a simple web application using Flask. The web application will interact with the ML model artifacts such that we can make predictions by giving input features values.
'''

'''
Importing the libraries.
'''

# Web app.
from flask import Flask, request, render_template

# Data manipulation.
import numpy as np
import pandas as pd

# File handling.
import os

# Predictions.
from src.pipeline.predict_pipeline import InputData, PredictPipeline


application = Flask(__name__)


app = application


# Route for the home page.

@app.route('/')
def index():
    '''
    Route handler for the home page.

    This function handles the GET request for the home page. It renders the 'index.html' template, which serves as the
    homepage for the customer credit risk prediction web application.

    :return: The rendered home page.
    :rtype: str
    '''
    return render_template('index.html')


# Route for prediction page.

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    '''
    Route handler for predicting credit risk.

    This function handles the POST request for predicting credit risk based on input data. If the request is a GET request,
    the function renders the 'home.html' template. If the request is a POST request, it collects input data from the form,
    processes it to make a prediction, and returns the prediction result.

    :return: The prediction result.
    :rtype: str
    '''

    if request.method == 'GET':
        return render_template('home.html')
    else:
        input_data = InputData(
            customer_age=request.form.get('customer_age'),
            gender=request.form.get('gender'),
            dependent_count=request.form.get('dependent_count'),
            education_level=request.form.get('education_level'),
            marital_status=request.form.get('marital_status'),
            income_category=request.form.get('income_category'),
            card_category=request.form.get('card_category'),
            months_on_book=request.form.get('months_on_book'),
            total_relationship_count=request.form.get('total_relationship_count'),
            months_inactive_12_mon=request.form.get('months_inactive_12_mon'),
            contacts_count_12_mon=request.form.get('contacts_count_12_mon'),
            credit_limit=request.form.get('credit_limit'),
            total_revolving_bal=request.form.get('total_revolving_bal'),
            total_amt_chng_q4_q1=request.form.get('total_amt_chng_q4_q1'),
            total_trans_amt=request.form.get('total_trans_amt'),
            total_trans_ct=request.form.get('total_trans_ct'),
            total_ct_chng_q4_q1=request.form.get('total_ct_chng_q4_q1'),
            avg_utilization_ratio=request.form.get('avg_utilization_ratio')
        )

        input_df = input_data.get_input_data_df()
        print(input_df)
        print('\nBefore prediction.')

        predict_pipeline = PredictPipeline()
        print('\nMid prediction')
        prediction = predict_pipeline.predict(input_df)
        print('\nAfter prediction.')

        return prediction
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)