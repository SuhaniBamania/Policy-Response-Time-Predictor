# evaluate_model.py

import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, r2_score


MODEL_PATH = 'model.pkl'
TEST_DATA_PATH = 'test_dataset.csv'


def evaluate_model():
    """
    Loads the model and test data, makes predictions, and prints
    evaluation metrics like Mean Absolute Error and R-squared score.
    """
    print("--- Starting Model Evaluation ---")
    

    try:
        model = joblib.load(MODEL_PATH)
        print(f"Successfully loaded model from '{MODEL_PATH}'.")
        
        df_test = pd.read_csv(TEST_DATA_PATH)
        print(f"Successfully loaded {len(df_test)} rows from '{TEST_DATA_PATH}'.")
    except FileNotFoundError as e:
        print(f"--- ERROR: File not found ---")
        print(e)
        print("Please make sure 'model.pkl' and 'test_dataset.csv' are in the same folder.")
        return
    
    X_test = df_test.drop(['resolution_days', 'request_id', 'filing_date'], axis=1)
    y_test = df_test['resolution_days'] 

    
    print("\nMaking predictions on the test set...")
    predictions = model.predict(X_test)
    print("Predictions complete.")

    
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n--- Model Performance on Test Set ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} days")
    print(f"This means, on average, the model's prediction is off by about {mae:.2f} days.")
    print("-" * 35)
    print(f"R-squared (R²) Score: {r2:.4f}")
    print(f"This means the model explains approximately {r2:.1%} of the variance in resolution times.")
    print("-----------------------------------")



if __name__ == '__main__':
    evaluate_model()
