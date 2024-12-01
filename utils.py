import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind, spearmanr, pointbiserialr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List

def numeric_statistics(data: pd.DataFrame, numeric_columns: List[str]):
    """
    This function calculates the mean, median, min, max, and standard deviation of a given set
    of numeric columns in a pandas DataFrame.
    """
    # Initialize an empty dictionary to store the statistics
    numeric_statistics = pd.DataFrame()
    for col in numeric_columns:
        col_data = data[col]
        val = [
            col,
            col_data.median(),
            col_data.mean(),
            col_data.std(),
            col_data.min(),
            col_data.quantile(0.05),
            col_data.quantile(0.1),
            col_data.quantile(0.15),
            col_data.quantile(0.2),
            col_data.quantile(0.25),
            col_data.quantile(0.5),
            col_data.quantile(0.75),
            col_data.quantile(0.8),
            col_data.quantile(0.85),
            col_data.quantile(0.9),
            col_data.quantile(0.95),
            col_data.quantile(0.99),
            col_data.max()]
        columns = ['Variable','Median','Mean','Std. Deviation','Min. Value','Percentile 5','Percentile 10','Percentile 15','Percentile 20','Percentile 25','Percentile 50','Percentile 75','Percentile 80','Percentile 85','Percentile 90','Percentile 95','Percentile 99','Max. value']
        table = pd.DataFrame([val], columns=columns)
        numeric_statistics = pd.concat([numeric_statistics, table], ignore_index=True)
    return numeric_statistics

# Categorical vs Categorical
def categorical_vs_categorical(df: pd.DataFrame,cat1:str, cat2:str):
    contingency_table = pd.crosstab(df[cat1], df[cat2])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    print(f"Chi-square Test between {cat1} and {cat2}: p-value = {p}")
    sns.heatmap(contingency_table, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Contingency Table: {cat1} vs {cat2}")
    plt.show()


# Categorical vs Numeric
def categorical_vs_numeric(df: pd.DataFrame,cat:str, num:str):
    sns.boxplot(x=cat, y=num, data=df)
    plt.title(f"Boxplot of {num} by {cat}")
    plt.show()
    for category in df[cat].unique():
        group = df[df[cat] == category][num]
        print(f"{cat}={category}: Mean={group.mean():.2f}, Std={group.std():.2f}")
    print('\n')

# Numeric vs Numeric
def numeric_vs_numeric(df:pd.DataFrame,num1:str, num2:str):
    correlation, p = spearmanr(df[num1], df[num2])
    print(f"Spearman Correlation between {num1} and {num2}: {correlation:.2f}, p-value={p:.2g}")
    sns.scatterplot(x=num1, y=num2, data=df)
    plt.title(f"Scatterplot of {num1} vs {num2}")
    plt.show()

# Relationship with Target
def relationship_with_target(df: pd.DataFrame):
    for col in df.columns:
        if col == "y":
            continue
        print(f"Analyzing {col} vs Target (y):")
        if col in ['job','marital','education','default','housing','loan','contact','month','poutcome']:
            categorical_vs_categorical(df,col, "y")
        else:
            categorical_vs_numeric(df,"y", col)

def cramers_v_matrix(df: pd.DataFrame, categorical_columns:List[str]):
    """
    Compute the Cramér's V matrix for all categorical variables in a dataset.
    
    Parameters:
    df (DataFrame): The dataset.
    categorical_columns (list): List of column names for categorical variables.
    
    Returns:
    DataFrame: A matrix showing the Cramér's V values.
    """
    n = len(categorical_columns)
    cramers_v = np.zeros((n, n))
    
    for i, var1 in enumerate(categorical_columns):
        for j, var2 in enumerate(categorical_columns):
            if i == j:
                cramers_v[i, j] = 1.0
            else:
                contingency_table = pd.crosstab(df[var1], df[var2])
                chi2, _, _, _ = chi2_contingency(contingency_table)
                n_samples = contingency_table.sum().sum()
                cramers_v[i, j] = np.sqrt(chi2 / (n_samples * (min(contingency_table.shape) - 1)))
    
    return pd.DataFrame(cramers_v, index=categorical_columns, columns=categorical_columns)


def point_biserial_matrix(df: pd.DataFrame, numerical_columns: List[str], target_column:str):
    """
    Compute the Point Biserial Correlation between numerical variables and a binary target variable.
    
    Parameters:
    df (DataFrame): The dataset.
    numerical_columns (list): List of column names for numerical variables.
    target_column (str): The binary target variable column name.
    
    Returns:
    DataFrame: A matrix showing the Point Biserial Correlation values.
    """
    results = {}
    for num_col in numerical_columns:
        if df[target_column].nunique() == 2:  # Ensure binary target variable
            y_binary = (df[target_column] == df[target_column].unique()[1]).astype(int)
            corr, _ = pointbiserialr(df[num_col], y_binary)
            results[num_col] = corr
        else:
            raise ValueError("The target variable must be binary.")
    return pd.DataFrame.from_dict(results, orient='index', columns=['Point Biserial Correlation'])

