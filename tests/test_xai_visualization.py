"""XAI Visualization Tests for CardioGuard-AI."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.xai.visualize import (
    plot_gradcam_heatmap,
    plot_lead_attention,
    plot_ecg_with_prediction,
    LEAD_NAMES,
)
from src.xai.shap_xgb import (
    plot_shap_summary,
    plot_shap_waterfall,
)
from src.pipeline.compare_models import optimize_ensemble_weight


class TestGradCAMVisualization:
    """Tests for Grad-CAM visualization functions."""
    
    def test_plot_gradcam_heatmap_shape(self, tmp_path):
        """Test Grad-CAM heatmap generation."""
        # Create dummy data
        signal = np.random.randn(12, 1000)  # 12 leads, 1000 samples
        cam = np.random.rand(1000)  # CAM values
        
        save_path = tmp_path / "gradcam_test.png"
        fig = plot_gradcam_heatmap(signal, cam, save_path=save_path)
        
        assert save_path.exists()
        assert fig is not None
    
    def test_plot_gradcam_handles_transposed_signal(self, tmp_path):
        """Test that function handles (T, 12) signal shape."""
        signal = np.random.randn(1000, 12)  # Transposed
        cam = np.random.rand(1000)
        
        save_path = tmp_path / "gradcam_transposed.png"
        fig = plot_gradcam_heatmap(signal, cam, save_path=save_path)
        
        assert save_path.exists()
    
    def test_plot_lead_attention(self, tmp_path):
        """Test per-lead attention bar chart."""
        # attention_scores should be (12,) for 12 leads
        attention_scores = np.random.rand(12)
        
        save_path = tmp_path / "lead_attention_test.png"
        fig = plot_lead_attention(attention_scores, output_path=save_path)
        
        assert save_path.exists()
        assert fig is not None
    
    def test_plot_ecg_with_prediction(self, tmp_path):
        """Test ECG plot with prediction result."""
        signal = np.random.randn(12, 1000)
        
        save_path = tmp_path / "ecg_pred_test.png"
        fig = plot_ecg_with_prediction(
            signal,
            prediction=0.85,
            true_label=1,
            save_path=save_path,
        )
        
        assert save_path.exists()
        assert fig is not None


class TestSHAPVisualization:
    """Tests for SHAP visualization functions."""
    
    def test_plot_shap_summary_bar(self, tmp_path):
        """Test SHAP summary bar plot."""
        n_samples, n_features = 100, 64
        shap_values = np.random.randn(n_samples, n_features)
        features = np.random.randn(n_samples, n_features)
        
        save_path = tmp_path / "shap_summary_test.png"
        fig = plot_shap_summary(
            shap_values,
            features,
            save_path=save_path,
            plot_type="bar",
        )
        
        assert save_path.exists()
        assert fig is not None
    
    def test_plot_shap_waterfall(self, tmp_path):
        """Test SHAP waterfall plot."""
        n_samples, n_features = 10, 64
        shap_values = np.random.randn(n_samples, n_features)
        features = np.random.randn(n_samples, n_features)
        base_value = 0.5
        
        save_path = tmp_path / "shap_waterfall_test.png"
        fig = plot_shap_waterfall(
            shap_values,
            base_value,
            sample_idx=0,
            features=features,
            save_path=save_path,
        )
        
        assert save_path.exists()
        assert fig is not None


class TestEnsembleOptimization:
    """Tests for ensemble weight optimization."""
    
    def test_optimize_returns_valid_alpha(self):
        """Test that optimization returns alpha in [0, 1]."""
        np.random.seed(42)
        n = 100
        y_true = np.random.randint(0, 2, n)
        p_cnn = np.random.rand(n)
        p_xgb = np.random.rand(n)
        
        best_alpha, best_score, all_scores = optimize_ensemble_weight(
            y_true, p_cnn, p_xgb, metric="roc_auc"
        )
        
        assert 0.0 <= best_alpha <= 1.0
        assert 0.0 <= best_score <= 1.0
        assert len(all_scores) == 21  # Default 21 alpha values
    
    def test_optimize_with_custom_range(self):
        """Test optimization with custom alpha range."""
        np.random.seed(42)
        n = 100
        y_true = np.random.randint(0, 2, n)
        p_cnn = np.random.rand(n)
        p_xgb = np.random.rand(n)
        
        alpha_range = np.linspace(0.3, 0.7, 5)
        best_alpha, _, all_scores = optimize_ensemble_weight(
            y_true, p_cnn, p_xgb, alpha_range=alpha_range
        )
        
        assert 0.3 <= best_alpha <= 0.7
        assert len(all_scores) == 5
    
    def test_optimize_different_metrics(self):
        """Test optimization with different metrics."""
        np.random.seed(42)
        n = 100
        y_true = np.random.randint(0, 2, n)
        p_cnn = np.random.rand(n)
        p_xgb = np.random.rand(n)
        
        for metric in ["roc_auc", "pr_auc", "f1"]:
            best_alpha, best_score, _ = optimize_ensemble_weight(
                y_true, p_cnn, p_xgb, metric=metric
            )
            assert 0.0 <= best_alpha <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
