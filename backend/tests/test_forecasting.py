import unittest

from backend.app.services.preprocessing import get_processed_data
from backend.app.services.forecasting import EnergyForecaster, calculate_metrics, get_forecast_results


class TestForecasting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = get_processed_data()

    def test_01_metrics_calculation(self):
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.1, 1.9, 3.2, 3.8]
        metrics = calculate_metrics(y_true, y_pred)
        
        self.assertIn('MAE', metrics)
        self.assertIn('RMSE', metrics)
        self.assertIn('MAPE', metrics)
        self.assertGreater(metrics['MAE'], 0)

    def test_02_forecaster_train_and_predict(self):
        forecaster = EnergyForecaster()
        meta = forecaster.train(self.df)
        self.assertIn('prophet_metrics', meta)
        self.assertIn('baseline_metrics', meta)
        
        predictions = forecaster.predict(self.df, horizon_hours=24)
        self.assertEqual(len(predictions), 24)
        self.assertIn('Timestamp', predictions[0])
        self.assertIn('Forecast_kWh', predictions[0])

    def test_03_get_forecast_results_wrapper(self):
        res = get_forecast_results(self.df, horizon_hours=48)
        self.assertEqual(res['horizon_hours'], 48)
        self.assertEqual(len(res['predictions']), 48)


if __name__ == '__main__':
    unittest.main()
