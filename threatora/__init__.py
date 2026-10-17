"""
Threatora: Advanced Cyber Attack & Intrusion Detection Framework
"""

__version__ = "1.0.0"
__author__ = "Mohammad Hasim"

from models.autoencoder import Autoencoder, ReconstructionEnergyWrapper
from models.capsnet import SpatialTemporalCapsNet, GTNLayer
from models.classifier import train_xgboost, evaluate_predictions
from pipelines.pipeline_unsw_nb15 import run_unsw_pipeline
from pipelines.pipeline_cicids2018 import run_cicids_pipeline

__all__ = [
    "Autoencoder",
    "ReconstructionEnergyWrapper",
    "SpatialTemporalCapsNet",
    "GTNLayer",
    "train_xgboost",
    "evaluate_predictions",
    "run_unsw_pipeline",
    "run_cicids_pipeline",
]

