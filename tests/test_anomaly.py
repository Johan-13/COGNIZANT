import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import get_processed_data
from src.anomaly_detection import detect_anomalies


class TestAnomalyDetection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = get_processed_data()

    def test_01_detect_anomalies_output_structure(self):
        res = detect_anomalies(self.df, z_threshold=2.0)
        self.assertIn('summary', res)
        self.assertIn('anomalies', res)
        self.assertIn('total_anomalies', res['summary'])

    def test_02_anomaly_severity_filtering(self):
        res_sensitive = detect_anomalies(self.df, z_threshold=1.5)
        res_strict = detect_anomalies(self.df, z_threshold=3.0)
        self.assertGreaterEqual(res_sensitive['summary']['total_anomalies'], res_strict['summary']['total_anomalies'])


if __name__ == '__main__':
    unittest.main()
