import pytest
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from cve_classification import predict_severity

@pytest.fixture
def trained_components():
    # Create a simple trained model for testing
    X_train = ["Buffer overflow", "SQL injection", "XSS vulnerability"]
    y_train = [0, 1, 2]  # Critical, High, Medium
    
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_vec = vectorizer.fit_transform(X_train)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, y_train)
    
    return model, vectorizer

def test_prediction(trained_components):
    model, vectorizer = trained_components
    test_text = "Buffer overflow in system X"
    
    X_test_vec = vectorizer.transform([test_text])
    prediction = model.predict(X_test_vec)
    
    assert prediction[0] in [0, 1, 2]  # Should be one of our classes
    assert isinstance(prediction[0], int)

def test_model_saving_loading(trained_components, tmp_path):
    model, vectorizer = trained_components
    
    # Save components
    model_path = tmp_path / "test_model.pkl"
    vectorizer_path = tmp_path / "test_vectorizer.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    # Load components
    loaded_model = joblib.load(model_path)
    loaded_vectorizer = joblib.load(vectorizer_path)
    
    # Test prediction with loaded components
    test_text = "SQL injection vulnerability"
    X_test_vec = loaded_vectorizer.transform([test_text])
    prediction = loaded_model.predict(X_test_vec)
    
    assert prediction[0] in [0, 1, 2]
