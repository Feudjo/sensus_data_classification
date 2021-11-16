# Script to train machine learning model.
import os
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import train_model
# Load data
data = pd.read_csv(os.path.join(os.getcwd(), 'starter', 'starter', 'census_copy.csv'))

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
    training=False,
    encoder=encoder,
    lb=lb
)

# Train and save the model.
clf= train_model(X_train, y_train)
print(os.getcwd())
# joblib.dump(clf, 'starter/model/clf_model.pkl')
# joblib.dump(encoder, 'starter/model/encoder.pkl')
# joblib.dump(lb, 'starter/model/lbibarizer.pkl')


