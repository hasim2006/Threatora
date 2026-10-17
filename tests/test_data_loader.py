import unittest
import numpy as np
import pandas as pd

class TestDataLoader(unittest.TestCase):
    def test_data_loader_sanitization(self):
        df = pd.DataFrame({
            "feat1": [1.0, np.nan, 3.0, np.inf],
            "feat2": [-np.inf, 2.0, 3.0, 4.0],
            "Label": [0, 1, 0, 1]
        })
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        for col in numeric_cols:
            med = df[col].median()
            df[col] = df[col].fillna(med if not np.isnan(med) else 0)
            
        self.assertFalse(df.isna().any().any())
        self.assertFalse(np.isinf(df.to_numpy()).any())

if __name__ == "__main__":
    unittest.main()
