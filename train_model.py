import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("custom_loan_data.csv")

# Convert categorical columns to numerical values
df["Employment_Status"] = df["Employment_Status"].map({"Full-Time": 1, "Part-Time": 2, "Self-Employed": 3, "Unemployed": 0})
df["Marital_Status"] = df["Marital_Status"].map({"Married": 1, "Single": 0})
df["Loan_Approved"] = df["Loan_Approved"].map({"Yes": 1, "No": 0})

# Define features and target variable
X = df.drop(columns=["Applicant_ID", "Loan_Approved"])
y = df["Loan_Approved"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(model, "loan_model.pkl")
print("Model saved as 'loan_model.pkl'.")
