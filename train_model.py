# train_model.py  – trains Random-Forest & XGBoost to predict Claim_Limit
# ---------------------------------------------------------------
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv("final_health_dataset.csv")        # ← your latest file

# 🔄 Encode categorical columns ----------------------------------
df["Gender"] = df["Gender"].map({"male": 0, "female": 1, "other": 2})
df["BMI_Category"] = df["BMI_Category"].map(
    {"underweight": 0, "normal": 1, "overweight": 2, "obese": 3}
)
df["Exercise_Frequency"] = df["Exercise_Frequency"].map(
    {"never": 0, "rarely": 1, "occasionally": 2,
     "frequently": 3, "daily": 4}
)
df["Fast_Food_Consumption"] = df["Fast_Food_Consumption"].map({"no": 0, "yes": 1})
df["Smoking"] = df["Smoking"].map({"no": 0, "yes": 1})

# 🔄 NEW: encode Insurance_type ----------------------------------
df["Insurance_type"] = df["Insurance_type"].map(
    {"Family": 0, "Personal": 1, "Senior Citizen": 2}
)

# ---------------------------------------------------------------
# 2. SELECT FEATURES & TARGET
# ---------------------------------------------------------------
FEATURE_COLS = [
    "Age", "Gender", "BMI_Category", "Exercise_Frequency",
    "Fast_Food_Consumption", "Smoking",
    "Temperature", "Systolic_BP", "Diastolic_BP",
    "Heart_Rate", "Device_Battery_Level",
    "Insurance_type"                # 🔄 added feature
]

TARGET_COL = "Claim_Limit"          # 🔄 changed target

# Drop rows that still lack Claim_Limit (safety net)
df = df.dropna(subset=[TARGET_COL])

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# ---------------------------------------------------------------
# 3. TRAIN / TEST SPLIT & SCALING
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=df["Insurance_type"]
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ---------------------------------------------------------------
# 4. MODEL TRAINING
# ---------------------------------------------------------------
rf_model = RandomForestRegressor(
    n_estimators=400, max_depth=None, min_samples_leaf=2,
    random_state=42, n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
rf_mae = mean_absolute_error(y_test, rf_model.predict(X_test_scaled))

xgb_model = XGBRegressor(
    n_estimators=600, learning_rate=0.03, max_depth=6,
    subsample=0.9, colsample_bytree=0.9,
    objective="reg:squarederror", random_state=42,
    tree_method="hist", verbosity=0
)
xgb_model.fit(X_train_scaled, y_train)
xgb_mae = mean_absolute_error(y_test, xgb_model.predict(X_test_scaled))

# ---------------------------------------------------------------
# 5. PICK BEST & SAVE
# ---------------------------------------------------------------
if rf_mae < xgb_mae:
    best_model, best_name, best_mae = rf_model, "Random Forest", rf_mae
else:
    best_model, best_name, best_mae = xgb_model, "XGBoost", xgb_mae

with open("claim_limit_best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Training complete.")
print(f"   Random Forest MAE : {rf_mae:.2f}")
print(f"   XGBoost      MAE : {xgb_mae:.2f}")
print(f"👉  Saved **{best_name}** (MAE {best_mae:.2f}) to claim_limit_best_model.pkl")
