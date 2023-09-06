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


def sns_plots(data, features, histplot=True, countplot=False,     
              barplot=False, barplot_y=None, boxplot=False, 
              boxplot_x=None, outliers=False, kde=False, 
              hue=None):
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

    Returns:
        None

    Raises:
        CustomException: If an error occurs during the plot generation.

    '''
    
    try:
        num_features = len(features)
        num_rows = num_features // 3 + (num_features % 3 > 0)  

        fig, axes = plt.subplots(num_rows, 3, figsize=(20, 5*num_rows))  

        for i, feature in enumerate(features):
            row = i // 3  
            col = i % 3  

            ax = axes[row, col] if num_rows > 1 else axes[col] 
            
            if countplot:
                sns.countplot(data=data, x=feature, hue=hue, ax=ax)
            elif barplot:
                sns.barplot(data=data, x=feature, y=barplot_y, hue=hue, ax=ax)
            elif boxplot:
                sns.boxplot(data=data, x=boxplot_x, y=feature, showfliers=outliers, ax=ax)
            elif outliers:
                sns.boxplot(data=data, x=feature, ax=ax)
            else:
                sns.histplot(data=data, x=feature, hue=hue, kde=kde, ax=ax)

            ax.set_title(feature)  
            ax.set_xlabel('')  
        
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


def categorical_plots(data, features, hue=None, orient='v', palette="Set2"):
    '''
    Create categorical count plots for a list of features in a DataFrame.

    This function generates categorical count plots for a specified list of features in a given DataFrame.
    The plots can be oriented vertically or horizontally, and a color palette can be applied for differentiation.

    Parameters:
    - data: DataFrame containing the data.
    - features: List of categorical features to plot.
    - hue: Optional categorical variable for color differentiation.
    - orient: Orientation of the count plots ('v' if vertical or 'h' if horizontal).
    - palette: Seaborn color palette to use for the plots.

    Returns:
    - None (displays the plots).

    Exceptions:
    - CustomException: Raised if an error occurs during plot generation.

    Example usage:
    ```
    categorical_plots(df, categorical_features, hue='some_hue_variable', orient='v', palette='Set2')
    ```
    '''
    
    try:
        sns.set(style="whitegrid")
        
        # Calculate the number of rows and columns for subplot arrangement
        num_features = len(features)
        num_rows = num_features // 2 + num_features % 2
        num_cols = 2 if num_features > 1 else 1
        
        # Create subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(22, 6*num_rows), gridspec_kw={'hspace': 0.25, 'wspace': 0.2})
        
        # Flatten the axes list for easy iteration
        axes = axes.flatten()
        
        for i, feature in enumerate(features):
            # Determine the current subplot
            ax = axes[i]
            
            # Create the count plot based on orientation and hue
            if orient == 'h':
                sns.countplot(y=feature, data=data, hue=hue, palette=palette, ax=ax)
                
                # Add counts at the top of the bars
                for p in ax.patches:
                    ax.annotate(f'{round(p.get_width())}', (p.get_width() + 1, p.get_y() + p.get_height() / 2),
                                ha='left', va='center')

                ax.set_title(feature)  
                ax.set_ylabel('')

            else:
                sns.countplot(x=feature, data=data, hue=hue, palette=palette, ax=ax)
                
                # Add counts at the top of the bars
                for p in ax.patches:
                    ax.annotate(f'{round(p.get_height())}', (p.get_x() + p.get_width() / 2, p.get_height() + 1),
                                ha='center', va='bottom')
            
                ax.set_title(feature)  
                ax.set_xlabel('')

        # Remove any empty subplots
        for i in range(num_features, len(axes)):
            fig.delaxes(axes[i])
        
        # Adjust layout and show the plots
        plt.tight_layout()
        plt.show()
    
    except Exception as e:
        raise CustomException(e, sys)
