# 🏥 ClaimWise -- ML for Health Insurance

ClaimWise is a machine learning-based web application that predicts the maximum claimable insurance amount based on user-provided health, lifestyle, and insurance information.

The application integrates trained machine learning models with a Flask backend to process user inputs, apply preprocessing and feature scaling, and generate predictions through a web interface.

## 💡 Features

- Web-based login interface
- Health and insurance information input form
- Machine learning-based claim limit prediction
- Categorical feature encoding and data preprocessing
- Feature scaling using a trained StandardScaler
- Real-time prediction through the Flask web application
- Display of predicted claim amount and risk level
- Insurance plan recommendations based on the prediction

## 🛠️ Tech Stack

- **Programming:** Python
- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Flask
- **Machine Learning:** scikit-learn, XGBoost
- **Models:** Random Forest Regressor, XGBoost Regressor
- **Data Processing:** Pandas, NumPy
- **Preprocessing:** StandardScaler
- **Model Serialization:** Pickle, Joblib

## 📊 Input Features

The model uses health, lifestyle, and insurance-related information including:

- Age
- Gender
- BMI Category
- Exercise Frequency
- Fast Food Consumption
- Smoking
- Body Temperature
- Systolic Blood Pressure
- Diastolic Blood Pressure
- Heart Rate
- Device Battery Level
- Insurance Type

## 🤖 Machine Learning Approach

ClaimWise uses regression techniques to predict the `Claim_Limit` based on the provided input features.

Two machine learning models are used:

### Random Forest Regressor

A Random Forest Regressor is trained using multiple decision trees to predict the claim limit.

### XGBoost Regressor

An XGBoost Regressor is trained using gradient boosting to predict the claim limit.

Both models are evaluated using **Mean Absolute Error (MAE)**. The model with the lower MAE is selected and saved for use by the Flask application.

## 🔄 System Workflow

```text
User
  ↓
Login Page
  ↓
Health & Insurance Information
  ↓
Flask Backend
  ↓
Data Preprocessing
  ↓
Feature Scaling
  ↓
Trained ML Model
  ↓
Predicted Claim Limit
  ↓
Risk Level & Insurance Recommendations
Prediction Process
The user logs into the application.
The user provides health, lifestyle, and insurance information.
The Flask backend receives the submitted information.
Categorical inputs are converted into numerical representations.
The input features are transformed using the saved scaler.
The processed data is passed to the trained regression model.
The model predicts the maximum claimable amount.
The application displays the prediction along with the calculated risk level and recommendations.
📁 Project Structure
insurance-claim-prediction/
│
├── app.py
├── main.py
├── model_predict.py
├── train_model.py
├── requirements.txt
│
├── claim_limit_best_model.pkl
├── healthcare_best_model.pkl
├── scaler.pkl
├── final_health_dataset.csv
│
├── templates/
│   ├── insurancecred.html
│   └── question.html
│
└── README.md
⚙️ Installation
1. Clone the repository
git clone https://github.com/ratnakaramsneha/insurance-claim-prediction.git
cd insurance-claim-prediction
2. Create a virtual environment
python -m venv venv

Activate it on Windows:

venv\Scripts\activate

On Linux/macOS:

source venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
▶️ Run the Application

Start the Flask application:

python app.py

Open the application in your browser:

http://127.0.0.1:5000/
🧪 Model Training

The train_model.py script is used to train and evaluate the machine learning models.

Run:

python train_model.py

The training workflow includes:

Loading the healthcare dataset.
Preprocessing the input features.
Encoding categorical variables.
Separating input features and the Claim_Limit target.
Splitting the data into training and testing sets.
Applying feature scaling.
Training Random Forest and XGBoost regression models.
Evaluating the models using Mean Absolute Error.
Selecting the better-performing model.
Saving the trained model and scaler for use by the Flask application.
🎯 Project Objective

The objective of ClaimWise is to demonstrate how machine learning can be integrated with a web application to analyze health and insurance-related information and provide a data-driven estimate of the maximum claimable amount.

🔮 Future Improvements
Improve model performance through hyperparameter tuning.
Expand the dataset with additional healthcare and insurance records.
Experiment with additional regression algorithms.
Add prediction confidence or probability-based insights.
Improve authentication and application security.
Add interactive data visualizations.
Deploy the application to a cloud platform.
