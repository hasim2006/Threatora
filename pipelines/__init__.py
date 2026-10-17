"""
Threatora Dataset Evaluation Pipelines Package
"""

from .pipeline_unsw_nb15 import run_unsw_pipeline
from .pipeline_cicids2018 import run_cicids_pipeline

__all__ = ["run_unsw_pipeline", "run_cicids_pipeline"]
