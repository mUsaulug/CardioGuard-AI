"""
Integration tests for Consistency Guard.

Tests that the guard is properly called in the inference pipeline
and returns expected consistency information.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import torch

from src.pipeline.inference.consistency_guard import (
    AgreementType,
    check_consistency,
)


class TestConsistencyGuardIntegration:
    """Integration tests for consistency guard in pipeline."""

    def test_predict_includes_consistency_when_binary_model_provided(self):
        """predict() should return consistency field when binary_model is provided."""
        from src.pipeline.inference.run_inference_superclass import predict
        
        # Create mock models
        mock_cnn_model = MagicMock()
        mock_cnn_model.return_value = torch.tensor([[0.8, 0.2, 0.1, 0.05]])  # MI high
        mock_cnn_model.backbone = MagicMock(return_value=torch.randn(1, 64))
        
        mock_binary_model = MagicMock()
        mock_binary_model.return_value = torch.tensor([[2.0]])  # sigmoid(2.0) ≈ 0.88
        
        mock_xgb_data = {"models": {}, "calibrators": {}, "scaler": None}
        thresholds = {"MI": 0.5, "STTC": 0.5, "CD": 0.5, "HYP": 0.5}
        
        # Create dummy signal
        signal = np.random.randn(12, 1000).astype(np.float32)
        device = torch.device("cpu")
        
        # Call predict with binary_model
        result = predict(
            signal=signal,
            cnn_model=mock_cnn_model,
            xgb_data=mock_xgb_data,
            thresholds=thresholds,
            localization_model=None,
            device=device,
            binary_model=mock_binary_model,
            explain=False,
        )
        
        # Verify consistency field exists and has expected structure
        assert "consistency" in result
        assert result["consistency"] is not None
        
        consistency = result["consistency"]
        assert "agreement" in consistency
        assert "triage_level" in consistency
        assert "superclass_mi_prob" in consistency
        assert "binary_mi_prob" in consistency
        assert "warnings" in consistency

    def test_predict_consistency_none_without_binary_model(self):
        """predict() should return consistency=None when binary_model is not provided."""
        from src.pipeline.inference.run_inference_superclass import predict
        
        # Create mock models
        mock_cnn_model = MagicMock()
        mock_cnn_model.return_value = torch.tensor([[0.8, 0.2, 0.1, 0.05]])
        mock_cnn_model.backbone = MagicMock(return_value=torch.randn(1, 64))
        
        mock_xgb_data = {"models": {}, "calibrators": {}, "scaler": None}
        thresholds = {"MI": 0.5, "STTC": 0.5, "CD": 0.5, "HYP": 0.5}
        
        signal = np.random.randn(12, 1000).astype(np.float32)
        device = torch.device("cpu")
        
        # Call predict WITHOUT binary_model
        result = predict(
            signal=signal,
            cnn_model=mock_cnn_model,
            xgb_data=mock_xgb_data,
            thresholds=thresholds,
            localization_model=None,
            device=device,
            binary_model=None,  # Explicit None
            explain=False,
        )
        
        # Verify consistency is None
        assert "consistency" in result
        assert result["consistency"] is None

    def test_consistency_agree_mi_scenario(self):
        """Test AGREE_MI scenario through pipeline."""
        from src.pipeline.inference.run_inference_superclass import predict
        
        # Both models detect MI
        mock_cnn_model = MagicMock()
        mock_cnn_model.return_value = torch.tensor([[0.9, 0.1, 0.1, 0.1]])  # MI = 0.9
        mock_cnn_model.backbone = MagicMock(return_value=torch.randn(1, 64))
        
        mock_binary_model = MagicMock()
        mock_binary_model.return_value = torch.tensor([[3.0]])  # sigmoid(3.0) ≈ 0.95
        
        mock_xgb_data = {"models": {}, "calibrators": {}, "scaler": None}
        thresholds = {"MI": 0.5, "STTC": 0.5, "CD": 0.5, "HYP": 0.5}
        
        signal = np.random.randn(12, 1000).astype(np.float32)
        device = torch.device("cpu")
        
        result = predict(
            signal=signal,
            cnn_model=mock_cnn_model,
            xgb_data=mock_xgb_data,
            thresholds=thresholds,
            localization_model=None,
            device=device,
            binary_model=mock_binary_model,
            explain=False,
        )
        
        consistency = result["consistency"]
        assert consistency["agreement"] == "AGREE_MI"
        assert consistency["triage_level"] == "HIGH"

    def test_consistency_disagree_type_1_scenario(self):
        """Test DISAGREE_TYPE_1: Superclass MI, Binary No."""
        from src.pipeline.inference.run_inference_superclass import predict
        
        # Superclass detects MI, binary doesn't
        mock_cnn_model = MagicMock()
        mock_cnn_model.return_value = torch.tensor([[0.9, 0.1, 0.1, 0.1]])  # MI high
        mock_cnn_model.backbone = MagicMock(return_value=torch.randn(1, 64))
        
        mock_binary_model = MagicMock()
        mock_binary_model.return_value = torch.tensor([[-2.0]])  # sigmoid(-2.0) ≈ 0.12
        
        mock_xgb_data = {"models": {}, "calibrators": {}, "scaler": None}
        thresholds = {"MI": 0.5, "STTC": 0.5, "CD": 0.5, "HYP": 0.5}
        
        signal = np.random.randn(12, 1000).astype(np.float32)
        device = torch.device("cpu")
        
        result = predict(
            signal=signal,
            cnn_model=mock_cnn_model,
            xgb_data=mock_xgb_data,
            thresholds=thresholds,
            localization_model=None,
            device=device,
            binary_model=mock_binary_model,
            explain=False,
        )
        
        consistency = result["consistency"]
        assert consistency["agreement"] == "DISAGREE_TYPE_1"
        assert consistency["triage_level"] == "REVIEW"
        assert len(consistency["warnings"]) > 0


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
