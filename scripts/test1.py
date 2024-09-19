import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Function to handle missing values
def handle_missing_values(df):
    # Fill missing CompetitionDistance with a large value (no nearby competition)
    df['CompetitionDistance'].fillna(df['CompetitionDistance'].max() + 1, inplace=True)
    
    # Fill missing CompetitionOpenSinceMonth and CompetitionOpenSinceYear with median values
    df['CompetitionOpenSinceMonth'].fillna(df['CompetitionOpenSinceMonth'].median(), inplace=True)
    df['CompetitionOpenSinceYear'].fillna(df['CompetitionOpenSinceYear'].median(), inplace=True)
    
    # Fill missing Promo2SinceWeek and Promo2SinceYear (no Promo2 participation)
    df['Promo2SinceWeek'].fillna(0, inplace=True)
    df['Promo2SinceYear'].fillna(0, inplace=True)
    
    return df

# Function to create new features based on competition and promo information
def feature_engineering(df):
    # Create a feature 'CompetitionOpenSince' as a measure of competition age in months
    df['CompetitionOpenSince'] = (df['Date'].dt.year - df['CompetitionOpenSinceYear']) * 12 + \
                                  (df['Date'].dt.month - df['CompetitionOpenSinceMonth'])
    df['CompetitionOpenSince'] = df['CompetitionOpenSince'].apply(lambda x: max(x, 0))  # Handle negative values

    # Create a feature for Promo2 duration
    df['Promo2Duration'] = (df['Date'].dt.year - df['Promo2SinceYear']) * 52 + \
                           (df['Date'].dt.week - df['Promo2SinceWeek'])
    df['Promo2Duration'] = df['Promo2Duration'].apply(lambda x: max(x, 0))  # Handle negative values

    return df

# Function to encode categorical variables
def encode_categorical(df):
    # Label encode categorical columns
    label_encoders = {}
    categorical_columns = ['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval']
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Handle NA values by converting to string
        label_encoders[col] = le
    
    return df, label_encoders

# Function to extract features from the 'Date' column
def extract_date_features(df):
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x in [6, 7] else 0)  # Saturday = 6, Sunday = 7
    return df

# Function to scale numerical features
def scale_numerical(df, numerical_columns):
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df, scaler

import pandas as pd
import numpy as np

# Function to handle outliers based on IQR method
def handle_outliers(df, columns, method='cap'):
    """
    Handles outliers using the IQR method.
    
    Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        columns (list): List of columns to check for outliers.
        method (str): The method to handle outliers: 'remove', 'cap', or 'impute'.
                      'remove' - removes outliers,
                      'cap' - caps outliers to the IQR boundaries,
                      'impute' - replaces outliers with the median.
    
    Returns:
        pd.DataFrame: The dataframe with outliers handled.
    """
    for col in columns:
        # Calculate Q1 (25th percentile) and Q3 (75th percentile) for the column
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier boundaries
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Detect outliers
        outliers_lower = df[col] < lower_bound
        outliers_upper = df[col] > upper_bound
        
        if method == 'remove':
            # Remove outliers
            df = df[~(outliers_lower | outliers_upper)]
        
        elif method == 'cap':
            # Cap outliers to the lower and upper bounds
            df.loc[outliers_lower, col] = lower_bound
            df.loc[outliers_upper, col] = upper_bound
        
        elif method == 'impute':
            # Replace outliers with the median of the column
            median_value = df[col].median()
            df.loc[outliers_lower | outliers_upper, col] = median_value
        
    return df

# Example usage:
# Assuming 'train_data' is your dataset, and we want to handle outliers in 'Sales', 'Customers', and 'CompetitionDistance'.
outlier_columns = ['Sales', 'Customers', 'CompetitionDistance']

# Choose the method: 'remove', 'cap', or 'impute'
# For example, we use 'cap' to cap outliers in these columns:
train_data = handle_outliers(train_data, outlier_columns, method='cap')


# Master preprocessing function
def preprocess_data(df):
    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Extract date-based features
    df = extract_date_features(df)
    
    # Encode categorical variables
    df, label_encoders = encode_categorical(df)

    # we want to handle outliers in 'Sales', 'Customers', and 'CompetitionDistance'
    outlier_columns = ['Sales', 'Customers', 'CompetitionDistance']
    df = handle_outliers(df, outlier_columns, method='remove')



    
    # Select numerical columns for scaling (excluding target variable 'Sales')
    numerical_columns = ['Customers', 'CompetitionDistance', 'CompetitionOpenSince', 'Promo2Duration']
    
    # Scale numerical columns
    df, scaler = scale_numerical(df, numerical_columns)
    
    return df, label_encoders, scaler


