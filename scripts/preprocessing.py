import pandas as pd 
import numpy as np
from datetime import datetime
import pandas as pd
from scipy import stats






def remove_outliers(df, threshold=0.01):
    """
    Remove outliers from the DataFrame based on Z-scores.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    threshold (float): The Z-score threshold to identify outliers.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    # Calculate Z-scores
    z_scores = stats.zscore(df.select_dtypes(include=['float64', 'int64']))
    
    # Create a boolean mask for outliers
    outliers_mask = (abs(z_scores) > threshold).all(axis=1)
    
    # Filter out the outliers
    df_cleaned = df[~outliers_mask]
    
    return df_cleaned

def replace_missing_with_mean(df, columns):
    for column in columns:
        # Handle infinite values
        df[column] = df[column].replace([np.inf, -np.inf], np.nan)
        # Impute missing values with mean
        df[column] = df[column].fillna(df[column].mean())
    return df

def replace_missing_with_median(df, columns):
    for column in columns:
        # Handle infinite values
        df[column] = df[column].replace([np.inf, -np.inf], np.nan)
        # Impute missing values with median
        df[column] = df[column].fillna(df[column].median())
    return df

def replace_missing_with_mode(df, columns):
    for column in columns:
        # Handle infinite values
        df[column] = df[column].replace([np.inf, -np.inf], np.nan)
        # Impute missing values with mode
        df[column] = df[column].fillna(df[column].mode().iloc[0])
    return df


def convert_vehicle_intro_date_to_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert VehicleIntroDate to VehicleAge by calculating the difference between the current year and the vehicle intro year.
    """
    # Ensure 'VehicleIntroDate' is a datetime column
    df['VehicleIntroDate'] = pd.to_datetime(df['VehicleIntroDate'], errors='coerce')
    
    # Get the current year
    current_year = datetime.now().year
    
    # Calculate vehicle age
    df['VehicleAge'] = current_year - df['VehicleIntroDate'].dt.year
    
    # Drop the original 'VehicleIntroDate' column (optional)
    df = df.drop(columns=['VehicleIntroDate'])
    
    return df


def count_missing_values(df):
    #count the missing values and return the percentage of the messing values 
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100
    return missing_count , missing_percentage

def missing_values(df):
    
        total_rows = df.shape[0]

        data = {'columns':[],
                'missing_values' : [],
                'missing_values_percentage':[]}
        missing = pd.DataFrame(data)

        # missing['columns']= df.columns
        missing['missing_values'] = df.isnull().sum()
        missing['missing_values_percentage'] = (missing['missing_values'] / total_rows) * 100
        return missing 


def replace_outliers_with_mean(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)

        # Replace outliers with the mean value of the column
        mean_value = df[col].mean()
        df.loc[outliers, col] = mean_value

    return df


def replace_outliers_with_percentile(df, columns, percentile=0.95):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)

        # Calculate the specified percentile value (e.g., 95th percentile)
        percentile_value = df[col].quantile(percentile)

        # Replace outliers with the percentile value
        df.loc[outliers, col] = percentile_value

    return df

def remove_null_rows(df, columns):
   
    # Drop rows where specified columns have null values
    cleaned_df = df.dropna(subset=columns)
    
    return cleaned_df



def count_outliers(df, columns, quartile=0.95):
   
    table = []
    
    for col in columns:
        # Calculate the 1st quartile (25%) and the 3rd quartile (75%)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        
        # Calculate the Interquartile Range (IQR)
        IQR = Q3 - Q1
        
        # Define outliers as values that are 1.5 * IQR above Q3 or below Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers above upper bound and below lower bound
        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        
        # Calculate the percentage of outliers
        outlier_percentage = (outlier_count / len(df)) * 100
        
        # Append the results to the table
        table.append([col, outlier_count, outlier_percentage])
    
    # Return the result as a DataFrame
    return pd.DataFrame(table, columns=['Column Name', 'Number of Outliers', 'Outlier Percentage'])
