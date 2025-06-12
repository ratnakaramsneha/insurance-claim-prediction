import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Load Dataset
df = pd.read_csv("full_health_dataset.csv")

# 2. Drop unused columns
df = df.drop(columns=["Patient_ID", "Timestamp", "Sensor_ID"])

# 3. Encode categorical columns for the target variable (Target_Health_Status)
label_encoder = LabelEncoder()
df["Target_Health_Status"] = label_encoder.fit_transform(df["Target_Health_Status"])

# 4. Separate features and target
X = df.drop(columns=["Target_Health_Status"])
y = df["Target_Health_Status"]

# 5. Apply One-Hot Encoding to categorical features like 'Sensor_Type' and any other categorical columns
# Update the transformer to include all categorical columns you need to encode
categorical_columns = X.select_dtypes(include=['object']).columns  # Automatically select categorical columns

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_columns)  # Apply OneHotEncoding to all categorical columns
    ],
    remainder='passthrough'  # Keep other (numerical) columns as is
)

# 6. Combine Preprocessing with Model (Pipeline)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),  # Apply scaling to all features after transformation
    ('model', RandomForestClassifier(random_state=42))  # Model
])

# 7. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Train Model
pipeline.fit(X_train, y_train)

# 9. Predict & Evaluate
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Healthy", "Unhealthy"])

# 10. Output Results
print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)
