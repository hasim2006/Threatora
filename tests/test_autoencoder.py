import unittest

try:
    import torch
    from models.autoencoder import Autoencoder, ReconstructionEnergyWrapper
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False

class TestAutoencoder(unittest.TestCase):
    def setUp(self):
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch is not available or missing C++ runtime dependencies")

    def test_autoencoder_forward(self):
        batch_size = 16
        input_dim = 40
        bottleneck_dim = 24
        
        model = Autoencoder(input_dim=input_dim, bottleneck_dim=bottleneck_dim)
        x = torch.randn(batch_size, input_dim)
        out = model(x)
        
        self.assertEqual(out.shape, (batch_size, input_dim))

    def test_reconstruction_energy(self):
        batch_size = 16
        input_dim = 40
        
        model = Autoencoder(input_dim=input_dim)
        energy_wrapper = ReconstructionEnergyWrapper(model)
        
        x = torch.randn(batch_size, input_dim)
        energy = energy_wrapper(x)
        
        self.assertEqual(energy.shape, (batch_size, 1))
        self.assertTrue(torch.all(energy >= 0.0))

if __name__ == "__main__":
    unittest.main()
