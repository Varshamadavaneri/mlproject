# Simple test for your ML script
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cre_classification import preprocess_data

def test_preprocess():
    """Test data preprocessing"""
    test_data = {"feature": [1, 2, 3]}
    result = preprocess_data(test_data)
    assert "feature" in result.columns
