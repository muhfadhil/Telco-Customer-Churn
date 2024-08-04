# Import libraries
import pandas as pd

# Load the data
df = pd.read_csv("../../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

"""
Data understanding

'customerID': Customer ID
'gender': Whether the customer is a male or a female
'SeniorCitizen': Whether the customer is a senior citizen or not (1, 0)
'Partner': Whether the customer has a partner or not (Yes, No)
'Dependents': Whether the customer has dependents or not (Yes, No)
'tenure': Number of months the customer has stayed with the company
'PhoneService': Whether the customer has a phone service or not (Yes, No)
'MultipleLines': Whether the customer has multiple lines or not (Yes, No, No phone service)
'InternetService': Customer’s internet service provider (DSL, Fiber optic, No)
'OnlineSecurity': Whether the customer has online security or not (Yes, No, No internet service)
'OnlineBackup': Whether the customer has online backup or not (Yes, No, No internet service)
'DeviceProtection': Whether the customer has device protection or not (Yes, No, No internet service)
'TechSupport': Whether the customer has tech support or not (Yes, No, No internet service)
'StreamingTV': Whether the customer has streaming TV or not (Yes, No, No internet service)
'StreamingMovies': Whether the customer has streaming movies or not (Yes, No, No internet service)
'Contract': The contract term of the customer (Month-to-month, One year, Two year)
'PaperlessBilling': Whether the customer has paperless billing or not (Yes, No)
'PaymentMethod': The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card
'MonthlyCharges': The amount charged to the customer monthly
'TotalCharges': The total amount charged to the customer
'Churn': Whether the customer churned or not (Yes or No)
"""

# Basic info
df.info()

# Look at first row data
df.iloc[0]

# Distribution for each columns
for col in df.columns:
    print(df[col].value_counts())
    print()

# Delete NaN or " " values for TotalCharges column
flag = df[["TotalCharges"]].apply(lambda x: x.str.contains(r"[^0-9.]"))

df = df[~flag["TotalCharges"]].reset_index(drop=True)

# Convert TotalCharges column to float dtype
df["TotalCharges"] = df["TotalCharges"].astype("float64")

# Check data duplicated
print(f"The number of data duplicated is {df.duplicated().sum()}")

# Drop customerID feature
df = df.drop("customerID", axis=1)

# Standardize feature names
df.columns = [col[0].upper() + col[1:] for col in df.columns]

# Export dataframe to pickle file
df.to_pickle("../../data/interim/01_custid_dropped.pkl")
