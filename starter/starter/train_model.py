# Script to train machine learning model.
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from .ml.data import process_data
from.ml.model import train_model

# Load data
data = pd.read_csv("census_copy.csv")

# Train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data.
X_test, y_test, encoder, lb = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# Train and save the model.
clf= train_model(X_train, y_train)
joblib.dump(clf, 'clf_model.pkl')

