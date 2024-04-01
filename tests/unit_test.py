import warnings
import unittest
import json
from api import app
import pandas as pd
import sklearn

class FlaskAppTest(unittest.TestCase):

    def setUp(self):
        app.testing = True
        self.app = app.test_client()    

    def test_home_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode('utf-8'), 'Welcome to the credit scoring api')

    def test_predict_endpoint(self):
        warnings.filterwarnings("ignore", category=UserWarning)

        df_test = pd.read_parquet('./data/df_test.parquet')
        df_test = df_test.sample(n=1).drop(['SK_ID_CURR','index'], axis=1)
        df_test = {'df_test': df_test.values.tolist()}
        headers = {'Content-Type': 'application/json'}
        
        response = self.app.post('/predict', data=json.dumps(df_test), headers=headers)
        data = json.loads(response.data.decode('utf-8'))
        print("Received data:", data)  # Add this line to print received data
        self.assertEqual(response.status_code, 200)
        self.assertTrue('prediction' in data)


if __name__ == '__main__':
    unittest.main(exit=False)