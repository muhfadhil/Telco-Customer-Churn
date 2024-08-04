# Import libraries
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_pickle("../../data/processed/02_cleaned.pkl")

# Split data into features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split into data training and data testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2, stratify=y
)

# Handling imbalance data
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Modeling
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf.score(X_train, y_train)
rf.score(X_test, y_test)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring="roc_auc",
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

best_model.score(X_train, y_train)
best_model.score(X_test, y_test)


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
        index=[None],
    )

    return result


train_eval = evaluate_model(best_model, X_train, y_train)
test_eval = evaluate_model(best_model, X_test, y_test)
