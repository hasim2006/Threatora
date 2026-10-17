import unittest

try:
    import torch
    from models.capsnet import GTNLayer, SpatialTemporalCapsNet
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False

class TestCapsNet(unittest.TestCase):
    def setUp(self):
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch is not available or missing C++ runtime dependencies")

    def test_gtn_layer_forward(self):
        batch_size = 16
        in_features = 20
        out_features = 32
        
        gtn = GTNLayer(in_features, out_features)
        x = torch.randn(batch_size, in_features)
        out = gtn(x)
        
        self.assertEqual(out.shape, (batch_size, out_features))

    def test_squash_function(self):
        capsnet = SpatialTemporalCapsNet(input_dim=20)
        s = torch.randn(10, 8, 8) * 10.0
        v = capsnet.squash(s)
        
        norms = torch.norm(v, dim=-1)
        self.assertTrue(torch.all(norms >= 0.0) and torch.all(norms < 1.0))

    def test_capsnet_feature_extraction(self):
        batch_size = 8
        input_dim = 20
        capsnet = SpatialTemporalCapsNet(input_dim=input_dim, num_capsules=8, cap_dim=8)
        
        x = torch.randn(batch_size, input_dim)
        features = capsnet.extract_features(x)
        
        self.assertEqual(features.shape, (batch_size, 64))

    def test_capsnet_forward(self):
        batch_size = 8
        input_dim = 20
        capsnet = SpatialTemporalCapsNet(input_dim=input_dim, num_classes=2)
        
        x = torch.randn(batch_size, input_dim)
        logits = capsnet(x)
        
        self.assertEqual(logits.shape, (batch_size, 2))

if __name__ == "__main__":
    unittest.main()
