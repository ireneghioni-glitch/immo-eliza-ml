import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# --- Constants ---------------------------------------------------------------------------------------

# --- StatBel for Feature Engineering ---
# https://statbel.fgov.be/en/open-data/population-statistical-sector-12
DEMOG_BE_DATASET = "../data/raw/StatBel/population_data/OPENDATA_SECTOREN_2025_NEW.txt"
# https://statbel.fgov.be/en/open-data/address-file-statistical-sector
NIS_CODES_DATASET = "../data/raw/StatBel/population_data/TF_RGN_ADDRESSES_STAT_SECTORS_20240401.txt"



# --- Functions ---------------------------------------------------------------------------------------


# ========== PREPROCESSING PHASES ==========


# --- Drop Columns ---

# use **kwargs for making it a flexible function for feature chenges on preprocessing
# config dictionary
def drop_useless_columns(df):

    df_copy = df.copy()

    COLS_TO_DROP = {
        "property_url": "Unique for each property - no predictive power",
        "address": "Unique for each property - no predictive power",
        "city": "Too many values - would affect negatively predictive power",
        "nearby_city": "Too many values - would affect negatively predictive power"
    }
    # train
    # cols_present = [c for c in COLS_TO_DROP if c in X_train.columns]
    # X_train = X_train.drop(columns=cols_present)
    # # test
    # cols_present = [c for c in COLS_TO_DROP if c in X_test.columns]
    # X_test = X_test.drop(columns=cols_present)

    return df_copy.drop(columns=COLS_TO_DROP, errors='ignore')


# --- Feature Engineering ---

def add_new_features(df):

    df_copy = df.copy()

    # ===========================
    #       'urban_density'
    # ===========================
    # add categorical column named `'urban_density'`, reporting the level of urban density 
    # (High, Medium, Low possible levels) according to corresponding `'postal_code'` of the 
    # property - I start from having it as a continuous numerical, by just calculating density;  
    # load and manipulation of StatBel datasets 
    demog_df = pd.read_csv(DEMOG_BE_DATASET, sep='|', encoding='cp1252')
    nis_df = pd.read_csv(NIS_CODES_DATASET, sep='|', encoding='cp1252')
    demog_df['OPPERVLAKKTE IN HM²'] = demog_df['OPPERVLAKKTE IN HM²'].str.replace(',', '.').astype(float)
    # add a new column to gave the info in KM² instead
    demog_df['surface_KM²'] = demog_df['OPPERVLAKKTE IN HM²'] / 100
    # calculations per comune
    comune_df = demog_df.groupby('CD_REFNIS').agg(
        tot_pop_per_comune=('TOTAL', 'sum'),
        tot_area_km2=('surface_KM²', 'sum')
    ).reset_index()
    # creating and adding 'urban_density'
    comune_df['urban_density'] = comune_df['tot_pop_per_comune'] / comune_df['tot_area_km2']
    bridge_df = nis_df[['CD_ZIP', 'CD_REFNIS']].drop_duplicates()
    df_zip_density = bridge_df.merge(comune_df[['CD_REFNIS', 'urban_density']], on='CD_REFNIS', how='inner')
    density_mapping = df_zip_density.groupby('CD_ZIP')['urban_density'].mean()
    # # train
    # X_train['urban_density'] = X_train['postal_code'].map(density_mapping)
    # # test
    # X_test['urban_density'] = X_test['postal_code'].map(density_mapping)
    df_copy['urban_density'] = df_copy['postal_code'].map(density_mapping)

    # ===============================
    #         'property_age'
    # ===============================
    # 2026 - df['building_year']
    # X_train['property_age'] = 2026 - X_train['building_year']
    # X_test['property_age'] = 2026 - X_test['building_year']
    df_copy['property_age'] = 2026 - df_copy['building_year']

    # ===============================
    #      'living_area_ratio'
    # ===============================
    # df['living_area_m2'] / df['total_area_m2']
    # X_train['living_area_ratio'] = X_train['living_area_m2'] / X_train['total_area_m2']
    # X_test['living_area_ratio'] = X_test['living_area_m2'] / X_test['total_area_m2']
    df_copy['living_area_ratio'] = df_copy['living_area_m2'] / df_copy['total_area_m2']

    # ===============================
    #        'bedroom_density'
    # ===============================
    # df['bedrooms'] / df['living_area_m2']
    # X_train['bedroom_density'] = X_train['bedrooms'] / X_train['living_area_m2']
    # X_test['bedroom_density'] = X_test['bedrooms'] / X_test['living_area_m2']
    df_copy['bedroom_density'] = df_copy['bedrooms'] / df_copy['living_area_m2']

    return df_copy


# --- Handle NaNs ---

def handle_missing_values(df):

    df_copy = df.copy()
    nan_prop = (df_copy.isnull().sum() / len(df_copy)) * 100
    # List of cols extraction
    # 1) columns with NaNs higher than 50%
    # 2) columns with NaNs lower than 5%
    # 3) columns with NaNs between 5% and 50%
    cols_middle = nan_prop[(nan_prop > 5) & (nan_prop < 50)].index.tolist()

    # 1)
    nan_to_0_cols_50 = ['parking_count', 'garden_area_m2']
    nan_to_median_cols = ['floors_total', 'floor_number']
    for col in nan_to_0_cols_50:
        # X_train[col] = X_train[col].fillna(0)
        # X_test[col] = X_test[col].fillna(0)
        df_copy[col] = df_copy[col].fillna(0)
    # 2) 
    # --- Latitude & Longitude ---
    # calculating median for latitude and for longitude grouping them by province on Train
    # to maintain geographical coherence
    lat_median_by_province = df_copy.groupby('province')['latitude'].median()
    lon_median_by_province = df_copy.groupby('province')['longitude'].median()
    # train
    # X_train['latitude'] = X_train['latitude'].fillna(X_train['province'].map(lat_median_by_province))
    # X_train['longitude'] = X_train['longitude'].fillna(X_train['province'].map(lon_median_by_province))
    # test
    # X_test['latitude'] = X_test['latitude'].fillna(X_test['province'].map(lat_median_by_province))
    # X_test['longitude'] = X_test['longitude'].fillna(X_test['province'].map(lon_median_by_province))
    df_copy['latitude'] = df_copy['latitude'].fillna(df_copy['province'].map(lat_median_by_province))
    df_copy['longitude'] = df_copy['longitude'].fillna(df_copy['province'].map(lon_median_by_province))
    # nan_to_median_cols list update
    nan_to_median_cols += ['bedrooms', 'urban_density', 'km_from_nearby_city', 'is_nearby_city_prestigious']
    # 3)
    nan_to_unknown_cols = df_copy.select_dtypes(exclude='number').columns
    # filtering columns in middle range to select the numerical ones only
    cols_middle_numerical = df_copy[cols_middle].select_dtypes(include='number').columns.tolist()
    # adding them to already existing nan_to_median_cols list
    nan_to_median_cols += cols_middle_numerical
    # NUMERIC
    for col in nan_to_median_cols:
        median_value = df_copy[col].median()
        # X_train[col] = X_train[col].fillna(median_value)
        # X_test[col] = X_test[col].fillna(median_value)
        df_copy[col] = df_copy[col].fillna(median_value)
    # CATEGORIAL
    # X_train[nan_to_unknown_cols] = X_train[nan_to_unknown_cols].fillna('Unknown')
    # X_test[nan_to_unknown_cols] = X_test[nan_to_unknown_cols].fillna('Unknown')
    df_copy[nan_to_unknown_cols] = df_copy[nan_to_unknown_cols].fillna('Unknown')

    return df_copy

def encode_features(df, ohe=None, ordinal=None, bins_density=None):

    df_copy = df.copy()

    # 'property_type'
    # replacing nominals with numbers
    type_mapping = {
        'House': 0,
        'Apartment': 1
    }
    # X_train['property_type'] = X_train['property_type'].str.capitalize().replace(type_mapping).astype("Int64")
    # X_test['property_type'] = X_test['property_type'].str.capitalize().replace(type_mapping).astype("Int64")
    df_copy['property_type'] = df_copy['property_type'].str.capitalize().replace(type_mapping).astype("Int64")

    # 'urban_density'
    # check numerical data distribution for optimal subdivision in 3 nominal categories
    # quantile banning
    # directly generating 0, 1 and 2 on Train (now modified on whole df)
    if bins_density is None:
        df_copy['urban_density'], bins_density = pd.qcut(
            df_copy['urban_density'],
            q=3,
            labels=False,
            retbins=True,
            duplicates='drop'
        )
    else:
        df_copy['urban_density'] = pd.cut(
            df_copy['urban_density'],
            bins=bins_density,
            labels=False,
            include_lowest=True,
        )
    # # apply same bins to Test
    # X_test['urban_density_encoded'] = pd.cut(
    #     X_test['urban_density'],
    #     bins=bins_density,
    #     labels=False,
    #     include_lowest=True
    # )

    # --- OneHotEncoder ---
    categ_cols = ['property_subtype', 'region', 'province']

    if ohe is None:
        ohe = OneHotEncoder(
            handle_unknown='ignore', 
            sparse_output=False)
            # sparse_output -> to make Pandas return a clean dataframe keeping columns original names
            # handle_unknown -> to gracefully handle unknown values that may be present in test set
        ohe.set_output(transform='pandas') # to be paired with sparse_output=False
        ohe_df = ohe.fit_transform(df_copy[categ_cols])

    else:
        ohe_df = ohe.transform(df_copy[categ_cols])
        
    df_copy = df_copy.drop(columns=categ_cols)
    df_copy = pd.concat([df_copy, ohe_df], axis=1)
    # # train
    # X_train_categ_encoded = encoder.fit_transform(X_train[categ_cols])
    # X_train = X_train.drop(columns=categ_cols)
    # X_train = pd.concat([X_train, X_train_categ_encoded], axis=1)
    # # test
    # X_test_categ_encoded = encoder.transform(X_test[categ_cols])
    # X_test = X_test.drop(columns=categ_cols)
    # X_test = pd.concat([X_test, X_test_categ_encoded], axis=1)

    # --- Ordinal Encoding ---
    # 'state_of_the_building'
    redundancies_mapping = {
        'Excellent': 'Fully renovated',
    }
    # X_train['state_of_the_building'] = X_train['state_of_the_building'].replace(redundancies_mapping)
    # X_test['state_of_the_building'] = X_test['state_of_the_building'].replace(redundancies_mapping)
    df_copy['state_of_the_building'] = df_copy['state_of_the_building'].replace(redundancies_mapping)
    # logical order
    order_state_of_the_building = [
        'To demolish', 
        'To restore', 
        'To renovate', 
        'Normal', 
        'Fully renovated', 
        'under construction', 
        'New'
    ]
    # 'kitchen_equipped'
    # logical order
    order_kitchen_equipped = [
        'Not equipped', 
        'Partially equipped', 
        'Fully equipped', 
        'Super equipped'
    ]
    # 'epc_score'
    # logical order with addictional categories
    order_epc_score = [
        'G', 
        'F', 
        'E-', 
        'E', 
        'E+', 
        'D-', 
        'D', 
        'D+', 
        'C-', 
        'C', 
        'C+', 
        'B-', 
        'B', 
        'B+', 
        'A-', 
        'A', 
        'A+', 
        'A++'
    ]
    # initialization
    cols_to_elaborate = ['state_of_the_building', 'kitchen_equipped', 'epc_score']

    if ordinal is None:
        ordinal = OrdinalEncoder(
            categories=[
                order_state_of_the_building, 
                order_kitchen_equipped, 
                order_epc_score],
            handle_unknown='use_encoded_value', 
            unknown_value=-1
        )
        ordinal.set_output(transform='pandas')
        # execution
        ordinal_df = ordinal.fit_transform(df_copy[cols_to_elaborate])
    
    else:
        ordinal_df = ordinal.transform(df_copy[cols_to_elaborate])
        
    df_copy = df_copy.drop(columns=cols_to_elaborate)
    df_copy = pd.concat([df_copy, ordinal_df], axis=1)

        # # train
        # X_train[cols_to_elaborate] = encoder.fit_transform(X_train[cols_to_elaborate])
        # # test
        # X_test[cols_to_elaborate] = encoder.transform(X_test[cols_to_elaborate])

    return df_copy, ohe, ordinal, bins_density


# --- Feature Scaling ---

def standardize_data(df, scaler=None):

    df_copy = df.copy()

    # other_num_cols = []
    # for col in df_copy.select_dtypes(include=[np.number]).columns:
    #     if df_copy[col].isin([0,1]).all() or df_copy[col].isin([0,1,2]).all():
    #         continue
    #     other_num_cols.append(col)

    cols_to_exclude = [
        'state_of_the_building', 
        'kitchen_equipped', 
        'epc_score', 
        'postal_code'
    ]
    cols_to_scale = [
        c for c in df_copy.select_dtypes(include=[np.number]).columns 
        if c not in cols_to_exclude]

    if not cols_to_scale:
        return df_copy, scaler
    
    if scaler is None:
        # Initialisation
        scaler = StandardScaler()
        # Execution
        df_copy[cols_to_scale] = scaler.fit_transform(df_copy[cols_to_scale])
    
    else:
        # Execution
        df_copy[cols_to_scale] = scaler.transform(df_copy[cols_to_scale])

    # # train
    # X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    # # test
    # X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    return df_copy, scaler


def remove_outliers(X_train, y_train):

    # --- Outliers Removal (1% lower and upper) ---
    # temporary reallignment of train sets
    train_temp = X_train.copy()
    train_temp['price'] = y_train.values
    # quantiles calculation
    q_low = train_temp['price'].quantile(0.01)
    q_high = train_temp['price'].quantile(0.99)
    # filtering
    train_filtered = train_temp[(train_temp['price'] > q_low) & (train_temp['price'] < q_high)]
    # assegnation of cleaned matrixes to feature_train and target_train
    X_train = train_filtered.drop(columns=['price'])
    y_train = train_filtered['price']

    return X_train, y_train


def are_there_strings(X_train):

    string_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    if string_columns:
        raise TypeError(
            f'❌ Preprocessing validation failed.\nThe following columns still contain string values:\n'
            f'\n{", ".join(string_columns)}.\nCheck again your encoding steps.'
        )
    else:
        return f"✅ Train data compliant. You can move on to model training."



