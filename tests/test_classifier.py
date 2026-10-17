import unittest
import numpy as np

try:
    from models.classifier import train_xgboost, evaluate_predictions
    CLASSIFIER_AVAILABLE = True
except (ImportError, OSError):
    CLASSIFIER_AVAILABLE = False

class TestClassifier(unittest.TestCase):
    def setUp(self):
        if not CLASSIFIER_AVAILABLE:
            self.skipTest("Classifier or dependencies not available in current environment")

    def test_xgboost_train_and_eval(self):
        np.random.seed(42)
        n_samples = 200
        n_features = 64
        
        X_train = np.random.randn(n_samples, n_features).astype(np.float32)
        y_train = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        
        X_test = np.random.randn(50, n_features).astype(np.float32)
        y_test = np.random.choice([0, 1], size=50, p=[0.7, 0.3])
        
        model, duration = train_xgboost(X_train, y_train, n_estimators=10, max_depth=3, lr=0.1, threads=2)
        self.assertGreater(duration, 0.0)
        
        metrics = evaluate_predictions(model, X_test, y_test)
        self.assertIn("acc", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertIn("cm", metrics)
        self.assertTrue(0.0 <= metrics["acc"] <= 1.0)

if __name__ == "__main__":
    unittest.main()
