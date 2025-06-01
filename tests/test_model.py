import unittest
import pandas as pd
from model_training import categorize_severity  # Assuming you moved the function to model_training.py

class TestSeverityCategorization(unittest.TestCase):
    def test_critical(self):
        self.assertEqual(categorize_severity(9.5), 'Critical')

    def test_high(self):
        self.assertEqual(categorize_severity(7.5), 'High')

    def test_medium(self):
        self.assertEqual(categorize_severity(5.0), 'Medium')

    def test_low(self):
        self.assertEqual(categorize_severity(2.5), 'Low')

if __name__ == '__main__':
    unittest.main()
