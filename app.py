from flask import Flask, request, render_template, redirect, url_for, jsonify
import joblib
import numpy as np



app = Flask(__name__)

# 🔄 Load the new model and scaler
model = joblib.load("claim_limit_best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Serve login page
@app.route('/')
def home():
    return render_template("insurancecred.html")

# Handle login submission
@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    if username == "user" and password == "1234":
        return render_template("question.html")
    else:
        error = "❌ Invalid username or password"
        return render_template("insurancecred.html", error=error)

# Handle prediction form submission
@app.route('/predict', methods=['POST'])
def predict():
    print("🧾 Received form data:")
    for k in request.form:
        print(f"{k} = {request.form[k]}")

    try:
        # Categorical mappings
        gender_map    = {'male': 0, 'female': 1, 'other': 2}
        bmi_map       = {'underweight': 0, 'normal': 1, 'overweight': 2, 'obese': 3}
        exercise_map  = {'never': 0, 'rarely': 1, 'occasionally': 2, 'frequently': 3, 'daily': 4}
        bool_map      = {'no': 0, 'yes': 1}
        insurance_map = {'family': 0, 'personal': 1, 'senior citizen': 2}

        # Extract and encode inputs
        age       = float(request.form['age'])
        gender    = gender_map[request.form['gender'].lower()]
        bmi_cat   = bmi_map[request.form['bmi_category'].lower()]
        ex_freq   = exercise_map[request.form['exercise_frequency'].lower()]
        fast_food = bool_map[request.form['fast_food'].lower()]
        smoking   = bool_map[request.form['smoking'].lower()]
        temp      = float(request.form['temp'])
        systolic  = float(request.form['systolic'])
        diastolic = float(request.form['diastolic'])
        hr        = float(request.form['heart_rate'])
        battery   = float(request.form['battery'])
        insurance_type = insurance_map[request.form['insurance_type'].lower()]

        # Combine all inputs into a feature vector
        input_data = np.array([[age, gender, bmi_cat, ex_freq, fast_food, smoking,
                                temp, systolic, diastolic, hr, battery, insurance_type]])
        scaled = scaler.transform(input_data)
        predicted_claim = model.predict(scaled)[0]

        return jsonify(predicted_claim_limit=round(float(predicted_claim), 2))

    except Exception as e:
        import traceback
        traceback.print_exc()  # logs full error trace to console
        return jsonify(error="Prediction failed", details=str(e)), 500


if __name__ == '__main__':
    app.run(debug=True)
