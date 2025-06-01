import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from cve_classification import train_test_split, vectorize_text

@pytest.fixture
def sample_text_data():
    return [
        "Buffer overflow vulnerability in X",
        "SQL injection in Y",
        "Cross-site scripting in Z",
        "Denial of service in A"
    ], [0, 1, 2, 0]  # 0: Critical, 1: High, 2: Medium

def test_train_test_split(sample_text_data):
    X, y = sample_text_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    assert len(X_train) == 3
    assert len(X_test) == 1
    assert len(y_train) == 3
    assert len(y_test) == 1

def test_vectorize_text(sample_text_data):
    X, _ = sample_text_data
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_vec = vectorizer.fit_transform(X)
    
    assert X_vec.shape[0] == len(X)
    assert X_vec.shape[1] <= 5000

def test_model_training(sample_text_data):
    X, y = sample_text_data
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_vec = vectorizer.fit_transform(X)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, y)
    
    assert hasattr(model, 'coef_')
    assert model.coef_.shape[0] == len(set(y))
