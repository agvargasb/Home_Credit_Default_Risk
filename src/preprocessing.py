from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    # 2. TODO Encode string categorical features (dytpe `object`):
    #     - If the feature has 2 categories encode using binary encoding,
    #       please use `sklearn.preprocessing.OrdinalEncoder()`. Only 4 columns
    #       from the dataset should have 2 categories.
    #     - If it has more than 2 categories, use one-hot encoding, please use
    #       `sklearn.preprocessing.OneHotEncoder()`. 12 columns
    #       from the dataset should have more than 2 categories.
    # Take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the OrdinalEncoder and
    #     OneHotEncoder classes, then use the fitted models to transform all the
    #     datasets.

    bi_categorical_features = ['NAME_CONTRACT_TYPE',
                               'FLAG_OWN_CAR',
                               'FLAG_OWN_REALTY',
                               'EMERGENCYSTATE_MODE']
    
    bi_categorical_transformer = Pipeline(
        steps=[
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1))
        ]
    )

    multi_categorical_features = ['CODE_GENDER',
                                  'NAME_TYPE_SUITE',
                                  'NAME_INCOME_TYPE',
                                  'NAME_EDUCATION_TYPE',
                                  'NAME_FAMILY_STATUS',
                                  'NAME_HOUSING_TYPE',
                                  'OCCUPATION_TYPE',
                                  'WEEKDAY_APPR_PROCESS_START',
                                  'ORGANIZATION_TYPE',
                                  'FONDKAPREMONT_MODE',
                                  'HOUSETYPE_MODE',
                                  'WALLSMATERIAL_MODE']

    multi_categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    # 3. TODO Impute values for all columns with missing data or, just all the columns.
    # Use median as imputing value. Please use sklearn.impute.SimpleImputer().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the SimpleImputer and then use the fitted
    #     model to transform all the datasets.


    # 4. TODO Feature scaling with Min-Max scaler. Apply this to all the columns.
    # Please use sklearn.preprocessing.MinMaxScaler().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the MinMaxScaler and then use the fitted
    #     model to transform all the datasets.

    numeric_features = list(set(working_train_df.columns)- set(bi_categorical_features + multi_categorical_features))

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler())
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("bi_cat", bi_categorical_transformer, bi_categorical_features),
            ("multi_cat", multi_categorical_transformer, multi_categorical_features)
        ]
    )

    preprocessor.fit(working_train_df)

    train = preprocessor.transform(working_train_df)
    val = preprocessor.transform(working_val_df)
    test = preprocessor.transform(working_test_df)

    return train, val, test


def preprocess_data_1(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    bi_categorical_features = ['NAME_CONTRACT_TYPE',
                               'FLAG_OWN_CAR',
                               'FLAG_OWN_REALTY',
                               'EMERGENCYSTATE_MODE']

    multi_categorical_features = ['CODE_GENDER',
                                  'NAME_TYPE_SUITE',
                                  'NAME_INCOME_TYPE',
                                  'NAME_EDUCATION_TYPE',
                                  'NAME_FAMILY_STATUS',
                                  'NAME_HOUSING_TYPE',
                                  'OCCUPATION_TYPE',
                                  'WEEKDAY_APPR_PROCESS_START',
                                  'ORGANIZATION_TYPE',
                                  'FONDKAPREMONT_MODE',
                                  'HOUSETYPE_MODE',
                                  'WALLSMATERIAL_MODE']
    
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1))
        ]
    )

    numeric_features = list(set(working_train_df.columns)- set(bi_categorical_features + multi_categorical_features))

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler())
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, bi_categorical_features + multi_categorical_features)
        ]
    )

    preprocessor.fit(working_train_df)

    train = preprocessor.transform(working_train_df)
    val = preprocessor.transform(working_val_df)
    test = preprocessor.transform(working_test_df)

    return train, val, test, preprocessor


def preprocess_data_2(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:

    print("Input train data shape: ", train_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    working_train_df = train_df.copy()
    working_test_df = test_df.copy()

    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    bi_categorical_features = ['NAME_CONTRACT_TYPE',
                               'FLAG_OWN_CAR',
                               'FLAG_OWN_REALTY',
                               'EMERGENCYSTATE_MODE']

    multi_categorical_features = ['CODE_GENDER',
                                  'NAME_TYPE_SUITE',
                                  'NAME_INCOME_TYPE',
                                  'NAME_EDUCATION_TYPE',
                                  'NAME_FAMILY_STATUS',
                                  'NAME_HOUSING_TYPE',
                                  'OCCUPATION_TYPE',
                                  'WEEKDAY_APPR_PROCESS_START',
                                  'ORGANIZATION_TYPE',
                                  'FONDKAPREMONT_MODE',
                                  'HOUSETYPE_MODE',
                                  'WALLSMATERIAL_MODE']
    
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1))
        ]
    )

    numeric_features = list(set(working_train_df.columns)- set(bi_categorical_features + multi_categorical_features))

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler())
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, bi_categorical_features + multi_categorical_features)
        ]
    )

    preprocessor.fit(working_train_df)

    train = preprocessor.transform(working_train_df)
    test = preprocessor.transform(working_test_df)

    return train, test, preprocessor
