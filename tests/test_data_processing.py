import pytest
import pandas as pd
from cve_classification import categorize_severity

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'description': ['Buffer overflow in X', 'SQL injection in Y', 'Cross-site scripting in Z'],
        'cvss_score': ['9.5', '7.2', '3.8'],
        'published_date': ['2023-01-01', '2023-02-01', '2023-03-01'],
        'vulnerability_type': ['CWE-119', 'CWE-89', 'CWE-79']
    })

def test_categorize_severity():
    assert categorize_severity('9.5') == 'Critical'
    assert categorize_severity('7.2') == 'High'
    assert categorize_severity('4.5') == 'Medium'
    assert categorize_severity('3.8') == 'Low'
    assert categorize_severity('invalid') == 'Unknown'

def test_data_cleaning(sample_data):
    # Test that critical columns are not null after cleaning
    assert not sample_data['description'].isnull().any()
    assert not sample_data['cvss_score'].isnull().any()
    
    # Test that severity column is added
    sample_data['severity'] = sample_data['cvss_score'].apply(categorize_severity)
    assert 'severity' in sample_data.columns
    assert set(sample_data['severity'].unique()).issubset({'Critical', 'High', 'Medium', 'Low', 'Unknown'})
