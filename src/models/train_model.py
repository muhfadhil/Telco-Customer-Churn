# Import libraries
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as imblearn_make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_pickle("../../data/processed/02_cleaned.pkl")

# Split data into features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

features_to_scale = ["Tenure", "Contract", "MonthlyCharges", "TotalCharges"]

preprocessor = make_column_transformer(
    (StandardScaler(), features_to_scale), remainder="passthrough"
)

pipeline = imblearn_make_pipeline(preprocessor, SMOTE(), RandomForestClassifier())

param_grid = {
    "randomforestclassifier__n_estimators": [100, 200, 300],
    "randomforestclassifier__max_depth": [None, 10, 20, 30],
    "randomforestclassifier__min_samples_split": [2, 5, 10],
    "randomforestclassifier__min_samples_leaf": [1, 2, 4],
}

# Split into data training and data testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2, stratify=y
)

# Modeling
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring="roc_auc",
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_


def evaluate_model(model, X, y):
    y_true = y
    y_pred = model.predict(X)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    result = pd.DataFrame(
        {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc,
        },
        index=["metric"],
    )

    return result


train_eval = evaluate_model(grid_search, X_train, y_train)
test_eval = evaluate_model(grid_search, X_test, y_test)
