import pytest
import pandas as pd

@pytest.fixture
def sample_cve_data():
    return pd.DataFrame({
        'description': [
            'Buffer overflow in X allows remote code execution',
            'SQL injection in Y allows database compromise',
            'Cross-site scripting in Z allows session hijacking'
        ],
        'cvss_score': ['9.8', '8.5', '5.4'],
        'published_date': ['2023-01-01', '2023-02-01', '2023-03-01'],
        'vulnerability_type': ['CWE-119', 'CWE-89', 'CWE-79']
    })
