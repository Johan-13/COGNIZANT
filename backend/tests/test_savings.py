import unittest

from backend.app.services.preprocessing import get_processed_data
from backend.app.services.savings import generate_recommendations, calculate_cost_and_savings


class TestSavings(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = get_processed_data()

    def test_01_generate_recommendations(self):
        recs = generate_recommendations(self.df)
        self.assertIsInstance(recs, list)
        self.assertGreater(len(recs), 0)
        self.assertIn('title', recs[0])
        self.assertIn('estimated_kwh_saving_monthly', recs[0])

    def test_02_calculate_cost_and_savings(self):
        res = calculate_cost_and_savings(monthly_kWh=500, tariff_rate=0.20)
        self.assertEqual(res['current_estimates']['flat_monthly_cost'], 100.0)
        self.assertEqual(res['current_estimates']['flat_yearly_cost'], 1200.0)
        self.assertEqual(len(res['savings_scenarios']), 5)
        # 10% scenario check: $10 saved monthly
        scenario_10 = [s for s in res['savings_scenarios'] if s['percentage'] == 10][0]
        self.assertEqual(scenario_10['saved_cost_monthly'], 10.0)


if __name__ == '__main__':
    unittest.main()
