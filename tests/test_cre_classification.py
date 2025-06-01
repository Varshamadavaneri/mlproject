import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cre_classification import preprocess_data, train_model

def test_preprocess_data():
    test_data = {"feature": [1, 2, 3]}
    result = preprocess_data(test_data)
    assert "feature" in result.columns

def test_train_model():
    X = [[1], [2], [3]]
    y = [0, 1, 0]
    model = train_model(X, y)
    assert hasattr(model, "predict")
