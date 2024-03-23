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

def analysis_plots(data, features, histplot=True, barplot=False, mean=None, text_y=0.5,    
                   outliers=False, boxplot=False, boxplot_x=None, kde=False, hue=None, 
                   nominal=False, color='#023047', figsize=(24, 12)):
    '''
    Generate plots for univariate and bivariate analysis.

    This function generates histograms, horizontal bar plots 
    and boxplots based on the provided data and features. 

    Args:
        data (DataFrame): The DataFrame containing the data to be visualized.
        features (list): A list of feature names to visualize.
        histplot (bool, optional): Generate histograms. Default is True.
        barplot (bool, optional): Generate horizontal bar plots. Default is False.
        mean (bool, optional): Generate mean bar plots of specified feature instead of proportion bar plots. Default is None.
        text_y (float, optional): Y coordinate for text on bar plots. Default is 0.5.
        outliers (bool, optional): Generate boxplots for outliers visualization. Default is False.
        boxplot (bool, optional): Generate boxplots for categories distributions comparison. Default is False.
        boxplot_x (str, optional): The feature to which the categories will have their distributions compared. Default is None.
        kde (bool, optional): Plot Kernel Density Estimate in histograms. Default is False.
        hue (str, optional): Hue for histogram and bar plots. Default is None.
        color (str, optional): The color of the plot. Default is '#023047'.
        figsize (tuple, optional): The figsize of the plot. Default is (24, 12).

    Returns:
        None

    Raises:
        CustomException: If an error occurs during the plot generation.

    '''
    
    try:
        # Get num_features and num_rows and iterating over the sublot dimensions.
        num_features = len(features)
        num_rows = num_features // 3 + (num_features % 3 > 0) 
        
        fig, axes = plt.subplots(num_rows, 3, figsize=figsize)  

        for i, feature in enumerate(features):
            row = i // 3  
            col = i % 3  

            ax = axes[row, col] if num_rows > 1 else axes[col] 
            
            if barplot:
                if mean:
                    data_grouped = data.groupby([feature])[[mean]].mean().reset_index()
                    data_grouped[mean] = round(data_grouped[mean], 2)
                    ax.barh(y=data_grouped[feature], width=data_grouped[mean], color=color)
                    for index, value in enumerate(data_grouped[mean]):
                        # Adjust the text position based on the width of the bars
                        ax.text(value + text_y, index, f'{value:.1f}', va='center', fontsize=15)
                else:
                    if hue:
                        data_grouped = data.groupby([feature])[[hue]].mean().reset_index().rename(columns={hue: 'pct'})
                        data_grouped['pct'] *= 100
                    else:
                        data_grouped = data.groupby([feature])[[feature]].count().rename(columns={feature: 'count'}).reset_index()
                        data_grouped['pct'] = data_grouped['count'] / data_grouped['count'].sum() * 100
        
                    ax.barh(y=data_grouped[feature], width=data_grouped['pct'], color=color)
                    
                    if pd.api.types.is_numeric_dtype(data_grouped[feature]):
                        ax.invert_yaxis()
                        
                    for index, value in enumerate(data_grouped['pct']):
                        # Adjust the text position based on the width of the bars
                        ax.text(value + text_y, index, f'{value:.1f}%', va='center', fontsize=15)
                
                ax.set_yticks(ticks=range(data_grouped[feature].nunique()), labels=data_grouped[feature].tolist(), fontsize=15)
                ax.get_xaxis().set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.grid(False)
        
            elif outliers:
                # Plot univariate boxplot.
                sns.boxplot(data=data, x=feature, ax=ax, color=color)
            
            elif boxplot:
                # Plot multivariate boxplot.
                sns.boxplot(data=data, x=boxplot_x, y=feature, showfliers=outliers, ax=ax, palette=palette)

            else:
                # Plot histplot.
                sns.histplot(data=data, x=feature, kde=kde, ax=ax, color=color, stat='proportion', hue=hue)

            ax.set_title(feature)  
            ax.set_xlabel('')  
        
        # Remove unused axes.
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
