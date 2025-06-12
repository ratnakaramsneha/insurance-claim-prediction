# 🏥 IoT Healthcare Health Status Prediction

This project uses a machine learning model to **predict a patient's health status** (`Healthy` or `Unhealthy`) using data collected from IoT healthcare sensors.

---

## 📂 Dataset
**File:** `healthcare_iot_target_dataset.csv`  
**Rows:** 200  
**Features include:**
- `Temperature (°C)`
- `Systolic_BP (mmHg)`
- `Diastolic_BP (mmHg)`
- `Heart_Rate (bpm)`
- `Device_Battery_Level (%)`
- `Battery_Level (%)`
- `Sensor_Type`, `Sensor_ID`, `Timestamp` (encoded)

**Target Variable:**
- `Target_Health_Status` → Binary classification: `Healthy` / `Unhealthy`

---

## 🧠 ML Model
The model uses a **Random Forest Classifier** with scikit-learn.

### 🧪 Workflow:
1. Load and clean the dataset
2. Encode categorical variables
3. Standardize features
4. Train/test split (80/20)
5. Model training and evaluation
6. Output:
   - Accuracy
   - Confusion Matrix
   - Classification Report

---

## 🛠 Dependencies

Install all dependencies with:

```bash
pip install -r requirements.txt
