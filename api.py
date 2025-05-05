from fastapi import FastAPI  #  Import FastAPI
app = FastAPI()  #  Create FastAPI instance
from fastapi import FastAPI
import joblib
import pandas as pd

# Load trained model
model = joblib.load("loan_model.pkl")

# Create FastAPI instance
app = FastAPI()

# Define a root endpoint
@app.get("/")
def home():
    return {"message": "Loan Sanctioning AI API is Running"}

# Define the prediction endpoint
@app.post("/predict/")
def predict_loan(income: int, loan_amount: int, credit_score: int, employment_status: str, existing_loans: int, marital_status: str):
    # Convert categorical values
    employment_map = {"Full-Time": 1, "Part-Time": 2, "Self-Employed": 3, "Unemployed": 0}
    marital_map = {"Married": 1, "Single": 0}

    if employment_status not in employment_map or marital_status not in marital_map:
        return {"error": "Invalid employment or marital status"}

    # Create input data frame
    input_data = pd.DataFrame([{
        "Income": income,
        "Loan_Amount": loan_amount,
        "Credit_Score": credit_score,
        "Employment_Status": employment_map[employment_status],
        "Existing_Loans": existing_loans,
        "Marital_Status": marital_map[marital_status]
    }])

    # Predict loan approval
    prediction = model.predict(input_data)
    result = "Approved" if prediction[0] == 1 else "Rejected"

    return {"Loan_Status": result}

