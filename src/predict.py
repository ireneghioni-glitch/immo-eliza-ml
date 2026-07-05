'''
========================================================================================================
predict.py - MODEL PREDICTIVE MODULE
========================================================================================================

'''

# --- Import libraries -----------------------------------------------------------------------------

import os
import joblib

# Data manipulation & visualization
import numpy as np
import pandas as pd

from utils.split_data import split_data
from utils.preprocessing import drop_useless_columns, add_new_features, handle_missing_values, encode_features, standardize_data, remove_outliers, are_there_strings
from utils.validation import calculate_metrics


# --- Constants ---------------------------------------------------------------------------------------

MODEL_DIR_PATH = "../models"
DEFAULT_MODEL_NAME = "xgboost_model.joblib"

FEAT_TEST_RAW_CLEAN = "../data/cleaned/features_test_raw_cleaned_for_predict.csv"


# --- Functions ---------------------------------------------------------------------------------------

def load_model(model_type="xgboost"):
    model_file_name = f"{model_type}_model.joblib"
    model_path = os.path.join(MODEL_DIR_PATH, model_file_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found in {model_path}.")
    
    print(f"Model loading in progress...")
    return joblib.load(model_path)

def predict_price(new_data, model, ohe, ordinal, scaler, bins_density):
    if isinstance(new_data, dict):
        df = pd.DataFrame([new_data])
    elif isinstance(new_data, pd.DataFrame):
        df = new_data.copy()
    else:
        raise TypeError(f"❌ Input data must be a Python dictionary or a Pandas DataFrame.")
    
    print(f"⏳ Preprocessing on {new_data} in progress...")
    # call preprocessing fuctions from utils.preprocessing
    X_live = df.copy()

    X_live = drop_useless_columns(X_live)
    X_live = add_new_features(X_live)
    X_live = handle_missing_values(X_live)
    X_live, _, _, _= encode_features(X_live, ohe=ohe, ordinal=ordinal, bins_density=bins_density)
    X_live, _ = standardize_data(X_live, scaler=scaler)

    print("🔮 The model is predicting the price...")
    predictions = model.predict(X_live)

    if len(predictions) == 1:
           return float(predictions[0])
    return predictions




if __name__ == "__main__":

    WINNING_MODEL = "xgboost"

    try:
        model = load_model(model_type=WINNING_MODEL)

        if os.path.exists(FEAT_TEST_RAW_CLEAN):
            test_df = pd.read_csv(FEAT_TEST_RAW_CLEAN)
            print(test_df.columns)
            print(test_df.head(3))
            # creation of a dummy record
            dummy_data = test_df.iloc[[0]]
            print("🎰 One row extracted from Raw Test set for simulation.")
        else:
            raise FileNotFoundError("File X_test_raw.csv not found: can't extract dummy data.")
        
        scaler = joblib.load(f"{MODEL_DIR_PATH}/scaler.joblib")
        ohe = joblib.load(f"{MODEL_DIR_PATH}/ohe.joblib")
        ordinal = joblib.load(f"{MODEL_DIR_PATH}/ordinal.joblib")
        bins_density = joblib.load(f"{MODEL_DIR_PATH}/bins_density.joblib")
        estimated_price = predict_price(dummy_data, model, ohe, ordinal, scaler, bins_density)

        print(f"🏠 Estimated price for dummy property: {estimated_price:,.2f} €.")
         
    except Exception as e:
        print(f"❌ Error: {e}.")