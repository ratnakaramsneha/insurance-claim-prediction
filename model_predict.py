"""
model_predict.py

This script loads the trained scaler and best model,
and provides a single prediction function: `predict_from_raw`.

This version predicts the **maximum claimable amount** (Claim_Limit),
not the insurance premium.
"""

import pickle
import pandas as pd

# Load the trained model and scaler
with open("claim_limit_best_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Must match train_model.py
FEATURE_COLS = [
    'Age', 'Gender', 'BMI_Category', 'Exercise_Frequency',
    'Fast_Food_Consumption', 'Smoking',
    'Temperature', 'Systolic_BP', 'Diastolic_BP',
    'Heart_Rate', 'Device_Battery_Level',
    'Insurance_type'
]

# Encoders
GENDER_MAP    = {'male': 0, 'female': 1, 'other': 2}
BMI_MAP       = {'underweight': 0, 'normal': 1, 'overweight': 2, 'obese': 3}
EXERCISE_MAP  = {'never': 0, 'rarely': 1, 'occasionally': 2, 'frequently': 3, 'daily': 4}
BOOL_MAP      = {'no': 0, 'yes': 1}
INS_TYPE_MAP  = {'family': 0, 'personal': 1, 'senior citizen': 2}

# Internal predictor (encoded only)
def _predict_claim_limit(encoded_input):
    input_df = pd.DataFrame([encoded_input], columns=FEATURE_COLS)
    X_scaled = scaler.transform(input_df)
    prediction = model.predict(X_scaled)
    return round(float(prediction[0]), 2)

# Public API function with raw inputs
def predict_from_raw(age, gender, bmi_category, exercise_frequency,
                     fast_food, smoking, temperature,
                     systolic, diastolic, heart_rate, battery,
                     insurance_type):
    """
    Predicts claim limit from raw user inputs (strings + numbers)
    """
    try:
        encoded = [
            age,
            GENDER_MAP[gender.lower()],
            BMI_MAP[bmi_category.lower()],
            EXERCISE_MAP[exercise_frequency.lower()],
            BOOL_MAP[fast_food.lower()],
            BOOL_MAP[smoking.lower()],
            temperature,
            systolic,
            diastolic,
            heart_rate,
            battery,
            INS_TYPE_MAP[insurance_type.lower()]
        ]
    except KeyError as e:
        raise ValueError(f"Invalid input value: {e}")

    return _predict_claim_limit(encoded)

# Manual test
if __name__ == "__main__":
    print("🔍 Test Prediction (raw input):")
    amount = predict_from_raw(
        age=62,
        gender='male',
        bmi_category='obese',
        exercise_frequency='rarely',
        fast_food='yes',
        smoking='yes',
        temperature=37.2,
        systolic=148,
        diastolic=92,
        heart_rate=88,
        battery=80,
        insurance_type='senior citizen'
    )
    print(f"Predicted Claimable Amount: ₹{amount}")
