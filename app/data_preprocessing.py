import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def exploratory_analysis(df: pd.DataFrame):
    print(df.head())

    print(df.describe())

    print(df.info())

    for column in df.select_dtypes(include=['object']).columns:
        print(f'Unique values for {column}:')
        print(df[column].value_counts())
        print('\n')

    df.hist(bins=15, figsize=(15, 10), layout=(4, 4))
    plt.suptitle('Histograms of Numerical Columns')
    plt.show()

    for column in df.select_dtypes(include=[np.number]).columns:
        plt.figure(figsize=(10, 4))
        sns.boxplot(x=df[column])
        plt.title(f'Boxplot of {column}')
        plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Heatmap of Correlation Between Features')
    plt.show()

    sns.pairplot(df.select_dtypes(include=[np.number]))
    plt.show()

    if 'label' in df.columns:
        for column in df.select_dtypes(include=['object']).columns:
            plt.figure(figsize=(10, 4))
            sns.countplot(x=column, hue='label', data=df)
            plt.title(f'Distribution of {column} by Label')
            plt.xticks(rotation=45)
            plt.show()

def clean_data(df):
    df = df.drop_duplicates()

    for column in df.columns:
        if df[column].dtype == np.number:
            df[column].fillna(df[column].median(), inplace=True)
        else:
            df[column].fillna(df[column].mode()[0], inplace=True)

    return df

def feature_engineering(df):
    df['debt_to_income_ratio'] = df['total_debt'] / df['monthly_income']


    selected_features = df[['age', 'monthly_income', 'debt_to_income_ratio', 'credit_score']]
    return selected_features

def preprocess_for_modeling(df):
    numeric_features = ['age', 'monthly_income', 'debt_to_income_ratio', 'credit_score']
    categorical_features = ['marital_status', 'employment_type']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)],
        remainder='drop')

    preprocessed_data = preprocessor.fit_transform(df)
    return preprocessed_data

def preprocess_data(df_path):
    df = pd.read_csv(df_path)

    df = clean_data(df)
    df = feature_engineering(df)
    preprocessed_data = preprocess_for_modeling(df)
    return preprocessed_data
