# train_model.py
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb # Using the better model
import joblib

# --- Configuration ---
TRAIN_DATA_PATH = 'train_dataset.csv'
MODEL_OUTPUT_PATH = 'model.pkl'

# --- Main Training Function ---
def train_model():
    print("--- Starting Model Training (with LightGBM) ---")

    try:
        df_train = pd.read_csv(TRAIN_DATA_PATH)
        print(f"Loaded {len(df_train)} rows from '{TRAIN_DATA_PATH}'.")
    except FileNotFoundError:
        print(f"ERROR: Training file not found at '{TRAIN_DATA_PATH}'.")
        return

    # Define features (X) and target (y) based on your actual CSV columns
    X = df_train.drop(['resolution_days', 'request_id', 'filing_date'], axis=1)
    y = df_train['resolution_days']

    # Define preprocessing for the categorical features
    categorical_features = [
        'request_type', 'department', 'category', 'state', 'region',
        'applicant_type', 'language', 'filing_day_of_week'
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough' # Keeps numerical columns like year, month
    )

    # Define the LightGBM model
    model = lgb.LGBMRegressor(random_state=42, n_jobs=-1)

    # Create the full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    print("\nTraining the model...")
    pipeline.fit(X, y)
    print("Model training complete.")

    print(f"\nSaving the trained model to '{MODEL_OUTPUT_PATH}'...")
    joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    print("Model saved successfully.")
    print("\n--- Training Process Finished ---")

if __name__ == '__main__':
    train_model()
