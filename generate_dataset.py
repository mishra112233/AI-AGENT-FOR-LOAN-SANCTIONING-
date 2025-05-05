import pandas as pd
import numpy as np
import random

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# Number of records
num_records = 1000

# Generate data
data = {
    "Applicant_ID": range(1001, 1001 + num_records),
    "Income": np.random.randint(200000, 2000000, num_records),
    "Loan_Amount": np.random.randint(50000, 1000000, num_records),
    "Credit_Score": np.random.randint(300, 900, num_records),
    "Employment_Status": np.random.choice(["Full-Time", "Part-Time", "Self-Employed", "Unemployed"], num_records),
    "Existing_Loans": np.random.randint(0, 5, num_records),
    "Marital_Status": np.random.choice(["Married", "Single"], num_records),
}

# Loan approval logic
def approve_loan(income, loan_amount, credit_score, employment_status, existing_loans):
    if credit_score > 700 and income > (loan_amount * 2) and employment_status != "Unemployed" and existing_loans < 3:
        return "Yes"
    return "No"

# Apply the function
data["Loan_Approved"] = [
    approve_loan(data["Income"][i], data["Loan_Amount"][i], data["Credit_Score"][i], data["Employment_Status"][i], data["Existing_Loans"][i])
    for i in range(num_records)
]

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

# Save dataset
df.to_csv("custom_loan_data.csv", index=False)

print("Dataset saved as 'custom_loan_data.csv'.")
