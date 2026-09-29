import os
import subprocess
import sys

import joblib
from sklearn import datasets
from sklearn.model_selection import train_test_split


def test_train_creates_outputs():
    # Run the training script and check it finishes without errors
    result = subprocess.run([sys.executable, "src/train.py"], capture_output=True, text=True)
    assert result.returncode == 0

    # Check if both output files were created
    assert os.path.exists("outputs/confusion_matrix.png")
    assert os.path.exists("outputs/model.joblib")


def test_accuracy_is_at_least_90_percent():
    # the same test split used in train.py
    iris = datasets.load_iris()
    _, X_test, _, y_test = train_test_split(iris.data, iris.target, random_state=42, test_size=0.2)

    # Load the saved model and check its accuracy
    model = joblib.load("outputs/model.joblib")
    assert model.score(X_test, y_test) >= 0.9