import unittest
import os
import pandas as pd

from backend.app.services.preprocessing import generate_sample_dataset, preprocess_data, get_processed_data, SAMPLE_DATA_PATH


class TestPreprocessing(unittest.TestCase):

    def test_01_generate_sample_dataset(self):
        temp_path = 'backend/app/data/test_temp_sample.txt'
        output_path = generate_sample_dataset(output_path=temp_path, days=10)
        self.assertTrue(os.path.exists(output_path))
        
        df_raw = pd.read_csv(output_path, sep=';')
        self.assertIn('Date', df_raw.columns)
        self.assertIn('Global_active_power', df_raw.columns)
        self.assertGreater(len(df_raw), 100)
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_02_preprocess_data(self):
        df_processed = preprocess_data(save_processed=False)
        self.assertIsInstance(df_processed, pd.DataFrame)
        self.assertIn('Energy_kWh', df_processed.columns)
        self.assertIn('Global_active_power_kW', df_processed.columns)
        self.assertIn('Occupancy_Level', df_processed.columns)
        self.assertEqual(df_processed.index.name, 'Datetime')
        self.assertFalse(df_processed['Energy_kWh'].isnull().any())
        self.assertTrue(set(df_processed['Occupancy_Level'].unique()).issubset({'Low', 'Medium', 'High'}))

    def test_03_get_processed_data_caching(self):
        df = get_processed_data(force_reprocess=False)
        self.assertGreater(len(df), 0)


if __name__ == '__main__':
    unittest.main()
