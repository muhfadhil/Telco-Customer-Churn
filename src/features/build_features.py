# Import libraries
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# Load the data
df = pd.read_pickle("../../data/interim/01_custid_dropped.pkl")

# Check missing value
df.isna().sum()

# Check cardinality for categorical columns
for col in df.select_dtypes(include="object").columns.to_list():
    print(f"Distribution of {col}")
    print(df[col].value_counts(normalize=True))
    print()

# Delete PhoneService column because "Yes" domination
del df["PhoneService"]

# Ordinal Encoding for Contract column because of hierarchy
ord_encoder = OrdinalEncoder(categories=[["Month-to-month", "One year", "Two year"]])
df["Contract"] = ord_encoder.fit_transform(df[["Contract"]])

# One Hot Encoding for other categorical columns
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
df = pd.get_dummies(df, dtype="int")


# Make function for above actions
def feature_engineer(data):
    # Delete PhoneService column because "Yes" domination
    del data["PhoneService"]

    # Ordinal Encoding for Contract column because of hierarchy
    ord_encoder = OrdinalEncoder(
        categories=[["Month-to-month", "One year", "Two year"]]
    )
    data["Contract"] = ord_encoder.fit_transform(data[["Contract"]])

    # One Hot Encoding for other categorical columns
    data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})
    data = pd.get_dummies(data, dtype="int")


df.to_pickle("../../data/processed/02_cleaned.pkl")
