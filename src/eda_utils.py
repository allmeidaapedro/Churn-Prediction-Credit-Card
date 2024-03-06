'''
This script aims to provide functions that will turn the exploratory data analysis (EDA) process easier. 
'''


'''
Importing libraries
'''

# Data manipulation and visualization.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Debugging.
from src.exception import CustomException
import sys

# Warnings.
from warnings import filterwarnings
filterwarnings('ignore')

palette=sns.color_palette(['#023047', '#e85d04', '#0077b6', '#ff8200', '#0096c7', '#ff9c33'])

def sns_plots(data, features, histplot=True, countplot=False,     
              barplot=False, barplot_y=None, boxplot=False, 
              boxplot_x=None, outliers=False, kde=False, 
              hue=None, color='#023047', palette=palette):
    '''
    Generate Seaborn plots for visualization.

    This function generates various types of Seaborn plots based on the provided
    data and features. Supported plot types include histograms, count plots,
    bar plots, box plots, and more.

    Args:
        data (DataFrame): The DataFrame containing the data to be visualized.
        features (list): A list of feature names to visualize.
        histplot (bool, optional): Generate histograms. Default is True.
        countplot (bool, optional): Generate count plots. Default is False.
        barplot (bool, optional): Generate bar plots. Default is False.
        barplot_y (str, optional): The name of the feature for the y-axis in bar plots.
        boxplot (bool, optional): Generate box plots. Default is False.
        boxplot_x (str, optional): The name of the feature for the x-axis in box plots.
        outliers (bool, optional): Show outliers in box plots. Default is False.
        kde (bool, optional): Plot Kernel Density Estimate in histograms. Default is False.
        hue (str, optional): The name of the feature to use for color grouping. Default is None.
        color (str, optional): The color for seaborn plots. Default is None.
        palete (str, option): The color palette for seaborn plots. Default is a custom palette.
    Returns:
        None

    Raises:
        CustomException: If an error occurs during the plot generation.

    '''
    
    try:
        # Getting num_features and num_rows and iterating over the sublot dimensions.
        num_features = len(features)
        num_rows = num_features // 3 + (num_features % 3 > 0)  

        fig, axes = plt.subplots(num_rows, 3, figsize=(20, 5*num_rows))  

        for i, feature in enumerate(features):
            row = i // 3  
            col = i % 3  

            ax = axes[row, col] if num_rows > 1 else axes[col] 
            
            if countplot:
                # Plotting countplot and adding the counts at the top of each bar.
                if hue:
                    sns.countplot(data=data, x=feature, hue=hue, ax=ax, palette=palette)
                else:
                    sns.countplot(data=data, x=feature, hue=hue, ax=ax, color=color)
                for container in ax.containers:
                    ax.bar_label(container)

            elif barplot:
                # Plotting barplot and adding the averages at the top of each bar.
                ax = sns.barplot(data=data, x=feature, y=barplot_y, hue=hue, ax=ax, ci=None, color=color)
                for container in ax.containers:
                    ax.bar_label(container)

            elif boxplot:
                # Plotting multivariate boxplot.
                sns.boxplot(data=data, x=boxplot_x, y=feature, showfliers=outliers, ax=ax, palette=palette)

            elif outliers:
                # Plotting univariate boxplot.
                sns.boxplot(data=data, x=feature, ax=ax, color=color)

            else:
                # Plotting histplot.
                sns.histplot(data=data, x=feature, hue=hue, kde=kde, ax=ax, color=color)

            ax.set_title(feature)  
            ax.set_xlabel('')  
        
        # Removing unused axes.
        if num_features < len(axes.flat):
            for j in range(num_features, len(axes.flat)):
                fig.delaxes(axes.flat[j])

        plt.tight_layout()
    
    except Exception as e:
        raise CustomException(e, sys)


def check_outliers(data, features):
    '''
    Check for outliers in the given dataset features.

    This function calculates and identifies outliers in the specified features
    using the Interquartile Range (IQR) method.

    Args:
        data (DataFrame): The DataFrame containing the data to check for outliers.
        features (list): A list of feature names to check for outliers.

    Returns:
        tuple: A tuple containing three elements:
            - outlier_indexes (dict): A dictionary mapping feature names to lists of outlier indexes.
            - outlier_counts (dict): A dictionary mapping feature names to the count of outliers.
            - total_outliers (int): The total count of outliers in the dataset.

    Raises:
        CustomException: If an error occurs while checking for outliers.

    '''
    
    try:
    
        outlier_counts = {}
        outlier_indexes = {}
        total_outliers = 0
        
        for feature in features:
            Q1 = data[feature].quantile(0.25)
            Q3 = data[feature].quantile(0.75)
            
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            feature_outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
            outlier_indexes[feature] = feature_outliers.index.tolist()
            outlier_count = len(feature_outliers)
            outlier_counts[feature] = outlier_count
            total_outliers += outlier_count
        
        print(f'There are {total_outliers} outliers in the dataset.')
        print()
        print(f'Number (percentage) of outliers per feature: ')
        print()
        for feature, count in outlier_counts.items():
            print(f'{feature}: {count} ({round(count/len(data)*100, 2)})%')

        return outlier_indexes, outlier_counts, total_outliers
    
    except Exception as e:
        raise CustomException(e, sys)
