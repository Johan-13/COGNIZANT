import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_01_summary_api(self):
        response = self.app.get('/api/summary')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('total_consumption_kWh', data['data'])

    def test_02_consumption_api(self):
        response = self.app.get('/api/consumption?period=daily&limit=30')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertIsInstance(data['data'], list)

    def test_03_forecast_api(self):
        response = self.app.get('/api/forecast?horizon=24')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertEqual(len(data['data']['predictions']), 24)

    def test_04_anomalies_api(self):
        response = self.app.get('/api/anomalies?threshold=2.0')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')

    def test_05_peaks_api(self):
        response = self.app.get('/api/peaks')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')

    def test_06_calculate_savings_api(self):
        response = self.app.post('/api/calculate-savings', json={'tariff_rate': 0.15, 'monthly_kWh': 450})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['data']['current_estimates']['monthly_cost'], 67.5)


if __name__ == '__main__':
    unittest.main()
