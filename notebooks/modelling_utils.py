'''
This script aims to provide functions that will turn the modelling process easier
'''

'''
Importing libraries
'''

# Data manipulation and visualization.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Modelling.
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, precision_recall_curve
import time

# Debugging.
from src.exception import CustomException
import sys

# Warnings.
from warnings import filterwarnings
filterwarnings('ignore')


def evaluate_models_cv(models, X_train, y_train):
    '''
    Evaluate multiple machine learning models using stratified k-fold cross-validation (the stratified k-fold is useful for dealing with target imbalancement).

    This function evaluates a dictionary of machine learning models by training each model on the provided training data
    and evaluating their performance using stratified k-fold cross-validation. The evaluation metric used is ROC-AUC score.

    Args:
        models (dict): A dictionary where the keys are model names and the values are instantiated machine learning model objects.
        X_train (array-like): The training feature data.
        y_train (array-like): The corresponding target labels for the training data.

    Returns:
        pd.DataFrame: A DataFrame containing the evaluation results for each model, including their average validation scores
                  and training scores.

    Raises:
        CustomException: If an error occurs while evaluating the models.

    '''


    try:
        # Stratified KFold in order to maintain the target proportion on each validation fold - dealing with imbalanced target.
        n_folds = 5
        stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        # Dictionaries with validation and training scores of each model for plotting further.
        models_val_scores = dict()
        models_train_scores = dict()

        for model in models:
            # Getting the model object from the key with his name.
            model_instance = models[model]

            # Measuring training time.
            start_time = time.time()
            
            # Fitting the model to the training data.
            model_instance.fit(X_train, y_train)

            end_time = time.time()
            training_time = end_time - start_time

            # Make predictions on training data and evaluate them.
            y_train_pred = model_instance.predict(X_train)
            train_score = roc_auc_score(y_train, y_train_pred)

            # Evaluate the model using k-fold cross validation, obtaining a robust measurement of its performance on unseen data.
            val_scores = cross_val_score(model_instance, X_train, y_train, scoring='roc_auc', cv=stratified_kfold)
            avg_val_score = val_scores.mean()
            val_score_std = val_scores.std()

            # Adding the model scores to the validation and training scores dictionaries.
            models_val_scores[model] = avg_val_score
            models_train_scores[model] = train_score

            # Printing the results.
            print(f'{model} results: ')
            print('-'*50)
            print(f'Training score: {train_score}')
            print(f'Average validation score: {avg_val_score}')
            print(f'Standard deviation: {val_score_std}')
            print(f'Training time: {round(training_time, 5)} seconds')
            print()

        # Plotting the results.
        print('Plotting the results: ')

        # Converting scores to a dataframe
        val_df = pd.DataFrame(list(models_val_scores.items()), columns=['Model', 'Average Val Score'])
        train_df = pd.DataFrame(list(models_train_scores.items()), columns=['Model', 'Train Score'])
        eval_df = val_df.merge(train_df, on='Model')

        # Plotting each model and their train and validation (average) scores.
        plt.figure(figsize=(15, 6))
        width = 0.35

        x = np.arange(len(eval_df['Model']))

        val_bars = plt.bar(x - width/2, eval_df['Average Val Score'], width, label='Average Validation Score', color='#66c2a5')
        train_bars = plt.bar(x + width/2, eval_df['Train Score'], width, label='Train Score', color='#fc8d62')

        plt.xlabel('Model')
        plt.ylabel('ROC-AUC Score')
        plt.title('Models Performances')
        plt.xticks(x, eval_df['Model'], rotation=45)

        # Add scores on top of each bar
        for bar in val_bars + train_bars:
            height = bar.get_height()
            plt.annotate('{}'.format(round(height, 2)),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

        plt.legend()
        plt.tight_layout()
        plt.show()

        return eval_df
    
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_classifier(y_true, y_pred, probas):
    '''
    Evaluate the performance of a binary classifier and visualize results.

    This function calculates and displays various evaluation metrics for a binary classifier,
    including the classification report, confusion matrix, and ROC AUC curve.

    Args:
    - y_true: True binary labels.
    - y_pred: Predicted binary labels.
    - probas: Predicted probabilities of positive class.

    Returns:
    - None (displays evaluation metrics).

    Raises:
    - CustomException: If an error occurs during evaluation.
    '''

    try:
        # Classification report
        print(classification_report(y_true, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot = True, fmt = 'd')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Values')
        plt.ylabel('Real Values')
        plt.show()
        
        # ROC AUC Curve and score
        fpr, tpr, thresholds = roc_curve(y_true, probas)
        auc = roc_auc_score(y_true, probas)

        plt.figure(figsize=(5, 3))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Random guessing line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()

    except Exception as e:
        raise CustomException(e, sys)
    

def plot_feature_importances(model, data):
    '''
    Plot feature importances of a given machine learning model.

    This function takes a trained machine learning model and the corresponding dataset used for training, and plots the
    feature importances of the model's attributes. Feature importances are sorted in descending order for visualization.

    Args:
        model (object): The trained machine learning model with a feature_importances_ attribute.
        data (DataFrame): The dataset containing the features used for training the model.

    Returns:
        None (displays feature importances).

    Raises:
        CustomException: If an error occurs while plotting feature importances.

    '''
    
    try:
        # Get feature importances
        importances = model.feature_importances_
        feature_names = data.columns 


        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        sorted_feature_names = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]

        # Plot feature importances
        color_sequence = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

        plt.figure(figsize=(12, 3))
        plt.title('Feature Importances')
        plt.bar(range(len(importances)), sorted_importances, tick_label=sorted_feature_names, color=color_sequence)
        plt.xticks(rotation=90)
        plt.show()

    except Exception as e:
        raise CustomException(e, sys)
    

def precision_vs_recall_curve(y_true, predicted_probas):
    '''
    Generate a precision-recall curve based on different threshold values.

    This function calculates precision and recall values for different threshold values,
    based on the model's estimated probability of being positive.

    Args:
        y_true (array-like): True binary labels of the data.
        predicted_probas (array-like): Predicted probabilities for the positive class.

    Returns:
        tuple: A tuple containing three elements:
            - precision (array): Precision values for different thresholds.
            - recall (array): Recall values for different thresholds.
            - threshold (array): Threshold values corresponding to precision and recall.

    Raises:
        CustomException: If an error occurs during curve generation.

    '''

    try:
        # Obtain precision and recall values (arrays) for different threshold
        # values, when compared to the model's estimated probability of being positive.
        precision, recall, threshold = precision_recall_curve(y_true, predicted_probas[:, 1])

        # Plotting precision-recall curve.
        plt.title('Precision vs Recall Curve')
        plt.plot(threshold, precision[:-1], 'b--', label='Precision')
        plt.plot(threshold, recall[:-1], 'r--', label='Recall')
        plt.xlabel('Threshold')
        plt.legend(loc='lower left')
        plt.ylim([0,1])

        return precision, recall, threshold

    except Exception as e:
        raise CustomException(e, sys)


def get_threshold_metrics(precision, recall, threshold, target_metric, target_metric_value):
    '''
    Get precision and recall metrics at a specified target metric value.

    This function retrieves precision and recall metrics at a specific target
    metric value for a given threshold. It is useful for selecting thresholds
    in classification problems.

    Args:
        precision (list): A list of precision values corresponding to different thresholds.
        recall (list): A list of recall values corresponding to different thresholds.
        threshold (list): A list of threshold values.
        target_metric (str): The target metric to consider ('recall' or 'precision').
        target_metric_value (float): The desired value of the target metric.

    Returns:
        tuple: A tuple containing three elements:
            - threshold_precision (float): Precision at the specified target metric value.
            - threshold_recall (float): Recall at the specified target metric value.
            - threshold_selected (float): The threshold value corresponding to the metrics.

    Raises:
        CustomException: If an error occurs while calculating the metrics.
    '''
    
    try:
        if target_metric == 'recall':
            recall_array = np.asarray(recall)
            target_value_index = np.where(recall_array[:-1] == target_metric_value)[0][0]

        else:
            precision_array = np.asarray(precision)
            target_value_index = np.where(precision_array[:-1] == target_metric_value)[0][0]
        
        threshold_precision = precision[target_value_index]
        threshold_recall = recall[target_value_index]
        threshold_selected = threshold[target_value_index]

        print(f'For a threshold of {threshold_selected}:')
        print(f'Recall: {threshold_recall}')
        print(f'Precision: {threshold_precision}')

        return threshold_precision, threshold_recall, threshold_selected

    except Exception as e:
        raise CustomException(e, sys)