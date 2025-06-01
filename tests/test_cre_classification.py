import pytest
from cre_classification import preprocess_data, train_model

def test_preprocess_empty_data():
    """Test empty input handling"""
    with pytest.raises(ValueError):
        preprocess_data({})

def test_model_prediction():
    """Test model training/prediction"""
    X_train = [[1], [2], [3]]
    y_train = [0, 1, 0]
    model = train_model(X_train, y_train)
    assert model.predict([[1.5]])[0] in [0, 1]  # Verify prediction format
