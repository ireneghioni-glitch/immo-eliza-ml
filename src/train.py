'''
========================================================================================================
train.py - MODEL TRAINING MODULE
========================================================================================================

This module contains:

    - 'load_split_preprocess_data()' function for loading original properties dataset to be splitted 
      and preprocessed, splitting and preprocessing original properties dataset, so it can be used for 
      model training and tersting - Train / Test split is done with 80 / 20 ratio on whole dataset in order
      to garantee validation, then transprmatons are applied on both of them (imputation and encoding) -
      preprocessing also includes removal of 1% lower and upper outliers;

    - 'train_model()' function that configures the model instance with optimized parameters. Requires model
      type as argument, to make it possible to train different models with same train.py module;

    - 'calculate_mestrics()' function for calculating model metrics, like R², MAE and RMSE (the one
      from metriics_utils.py in src/utils);

    - 'validate_model()' function that executes .predict() on Test set and calls 'calculate_metrics()' to
      print model final metrics as quality control measure;
    
    - 'save_model()' fuction that saves trained and validated model in models directory using joblib.dump().
'''


# --- Import libraries -----------------------------------------------------------------------------

import os
import joblib

# Data manipulation & visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
# other sklearn imports
from sklearn.base import BaseEstimator, TransformerMixin

import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Reproducibility - alwas se a random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# import functions from utils
from utils.split_data import split_data
from utils.preprocessing import drop_useless_columns, add_new_features, handle_missing_values, encode_features, standardize_data, remove_outliers, are_there_strings
from utils.validation import calculate_metrics, plot_predict_vs_actual


# --- Constants ---------------------------------------------------------------------------------------

# --- Properties Data ---
# parsed properties dataset path
DATASET_CSV_PATH = "../data/raw/properties_data.csv"
# features train/test sets paths
X_TRAIN_RAW_CSV_PATH = "../data/raw/train_test/X_train_raw.csv"
X_TEST_RAW_CSV_PATH = "../data/raw/train_test/X_test_raw.csv"
# target train/test sets paths
Y_TRAIN_CSV_PATH = "../../data/raw/train_test/y_train.csv"
Y_TEST_CSV_PATH = "../../data/raw/train_test/y_test.csv"
# feature train/test cleaned sets paths (ready for model training)
PROP_TRAIN_PREPROC_CSV_PATH = "../data/cleaned/properties_train_preprocessed.csv"
PROP_TEST_PREPROC_CSV_PATH = "../data/cleaned/properties_test_preprocessed.csv"

FEAT_TEST_RAW_CLEAN = "../data/cleaned/features_test_raw_cleaned_for_predict.csv"

# path to generated charts directory 
CHARTS_DIR_PATH = "../images/charts"
# path to generated plots directory 
PLOTS_DIR_PATH = "../images/plots"

# models_dir
MODEL_DIR_PATH = "../models"

# --- StatBel for Feature Engineering ---
# https://statbel.fgov.be/en/open-data/population-statistical-sector-12
DEMOG_BE_DATASET = "../data/raw/StatBel/population_data/OPENDATA_SECTOREN_2025_NEW.txt"
# https://statbel.fgov.be/en/open-data/address-file-statistical-sector
NIS_CODES_DATASET = "../data/raw/StatBel/population_data/TF_RGN_ADDRESSES_STAT_SECTORS_20240401.txt"


# --- Functions ---------------------------------------------------------------------------------------

# ========== LOAD, SPLIT AND PREPROCESS DATA FOR MODEL ==========

def load_split_preprocess_data():

    # --- 1. Loading --------------------------------------------

    # loading dataset
    properties = pd.read_csv(DATASET_CSV_PATH)

    # --- 2. Splitting ------------------------------------------
    features_train, features_test, target_train, target_test = split_data(properties)
    

    # --- 3. Preprocessing (Best Practice) ----------------------

    # --- Fix Wrong Dtypes ---
    # make 'is_nearby_city_prestigious' dtype Int64 rather than float64
    features_train['is_nearby_city_prestigious'] = features_train['is_nearby_city_prestigious'].astype('Int64')
    features_test['is_nearby_city_prestigious'] = features_test['is_nearby_city_prestigious'].astype('Int64')

    # PREPROCESS ON TRAIN SET AND CREATE SCALER
    
    # --- Drop Columns ---
    features_train = drop_useless_columns(features_train)
    # --- Feature Engineering ---
    features_train = add_new_features(features_train)
    # --- Handling Missing Values ---
    features_train = handle_missing_values(features_train)
    # --- Encoding ---
    features_train, ohe, ordinal, bins_density = encode_features(features_train, ohe=None, ordinal=None, bins_density=None)
    # --- Feature Scaling ---
    features_train, scaler = standardize_data(features_train, scaler=None)

    # PREPROCESS ON TEST SET USING SAME SCALER

    # --- Drop Columns ---
    features_test = drop_useless_columns(features_test)
    # --- Feature Engineering ---
    features_test = add_new_features(features_test)
    # --- Handling Missing Values ---
    features_test = handle_missing_values(features_test)

    # save for predict_price() in predict.py
    features_test.to_csv(FEAT_TEST_RAW_CLEAN, index=False)

    # --- Encoding ---
    features_test, _, _ , _= encode_features(features_test, ohe=ohe, ordinal=ordinal, bins_density=bins_density)
    # --- Feature Scaling ---
    features_test, _ = standardize_data(features_test, scaler=scaler)

    # --- Remove Outliers ONLY ON TRAIN SET ---
    features_train, target_train = remove_outliers(features_train, target_train)

    # data at this point should be cleaned and ready for the model
    # --- Safety Measure against eventually remaining Features containing Strings ---
    are_there_strings(features_train)
    are_there_strings(features_test)
    
    # --- Save cleaned Features train/test sets to CSV ---
    features_train.to_csv(PROP_TRAIN_PREPROC_CSV_PATH, index=False)
    features_test.to_csv(PROP_TEST_PREPROC_CSV_PATH, index=False)

    return features_train, features_test, target_train, target_test, ohe, ordinal, scaler, bins_density, properties


# ========== TRAIN MODEL ==========

def train_model(features_train, target_train, model_type):

    # winning parameters found in hyperparameter tuning notebook
    winning_learning_rate = 0.07
    winning_max_depth = 5
    winning_min_child_weight = 6
    winning_n_estimators = 600

    # model initialization
    if model_type == "xgboost":
        model = XGBRegressor(
            n_estimators=winning_n_estimators, 
            learning_rate=winning_learning_rate,
            max_depth=winning_max_depth,
            min_child_weight=winning_min_child_weight,
            random_state=42,
            verbosity=1
        )
    elif model_type == "random_forest":
        model = RandomForestRegressor(
           n_estimators=winning_n_estimators,
           random_state=42
        )
    else:
        raise ValueError(
            f"❌ {model_type} model not supported.\nChoose between 'xgboost' and 'random_forest'."
        )

    # model training with fit    
    print("🚀 Training the model...")
    model.fit(features_train, target_train)
    print(f"✅ Training of {model_type} model successfully completed.")

    return model


# ========== VALIDATE MODEL ==========

def validate_model(model, feature_test, target_test):

    print(f"Start of predictions generaton on Test Set for quality control...")
    predictions_euro = model.predict(feature_test)
    metrics = calculate_metrics(target_test, predictions_euro)

    print("#\n====================================================")
    print("🎯 MODEL PERFORMANCE METRICS ON TEST SET:")
    print("#\n====================================================")
    print(f"    R² Score (Optimized):   {metrics["R²"]}")
    print(f"    MAE (Mean Error):       {metrics["MAE"]}")
    print(f"    RMSE:                   {metrics["RMSE"]}")
    print("#\n====================================================")

    # scatterplot Predict vs Actual
    plot_predict_vs_actual(target_test, predictions_euro)


# ========== SAVE MODEL ==========

def save_model(model, model_type, scaler=None):
    model_dir = MODEL_DIR_PATH
    os.makedirs(model_dir, exist_ok=True)

    # save model
    model_file_name = f"{model_type}_model.joblib"
    model_path = os.path.join(model_dir, model_file_name)
    # save scaler
    if scaler is not None:
        joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))

    print(f"💾 Saving of model artifact in progress...")
    joblib.dump(model, model_path)
    print(f"✅ Model successfully saved in {model_path}.")




    
if __name__ == "__main__":
    MODEL_CHOICE = "xgboost"

    # 1. Loading and Transformation Pipeline (outliers removal included)
    X_train, X_test, y_train, y_test, trained_ohe, trained_ordinal, saved_scaler, bins_density, properties = load_split_preprocess_data()

    # 2. Model training on cleaned Train Set
    model = train_model(
        features_train=X_train, 
        target_train=y_train, 
        model_type=MODEL_CHOICE
    )
    
    # 3. Validation on Test Set (which remained intact)
    validate_model(
        model=model, 
        feature_test=X_test, 
        target_test=y_test
    )
    
    # 4. Serialization of final artifact
    save_model(
        model=model, 
        model_type=MODEL_CHOICE
    ) 

    # 5. Save estimators (Ohe, Ordinal, Scaler) in same directory to be used in predict.py
    scaler_file_name = "scaler.joblib"
    ohe_file_name = "ohe.joblib"
    ordinal_file_name = "ordinal.joblib"
    bins_density_file_name = "bins_density.joblib"
    joblib.dump(saved_scaler, os.path.join(MODEL_DIR_PATH, scaler_file_name))
    print(f"✅ Production scaler successfully saved in {MODEL_DIR_PATH} as {scaler_file_name}.")
    joblib.dump(trained_ohe, os.path.join(MODEL_DIR_PATH, ohe_file_name))
    print(f"✅ Production OneHot encoder successfully saved in {MODEL_DIR_PATH} as {ohe_file_name}.")
    joblib.dump(trained_ordinal, os.path.join(MODEL_DIR_PATH, ordinal_file_name))
    print(f"✅ Production Ordinal encoder successfully saved in {MODEL_DIR_PATH} as {ordinal_file_name}.")
    joblib.dump(bins_density, os.path.join(MODEL_DIR_PATH, bins_density_file_name))
    print(f"✅ Production bins density encoder successfully saved in {MODEL_DIR_PATH} as {bins_density_file_name}.")
