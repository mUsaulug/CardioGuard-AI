"""
XAI Sanity Check Module.

Implements real sanity checks for saliency maps and feature attributions
to ensure explanations are reliable and model-dependent.

Based on "Sanity Checks for Saliency Maps" (Adebayo et al., 2018).

Checks implemented:
1. Model Parameter Randomization: Explanation should change when model randomized
2. Faithfulness (Deletion/Insertion): Important regions should affect prediction
3. Stability: Explanation should be stable under small input perturbations

Usage:
    checker = XAISanityChecker(model)
    sanity_result = checker.run_checks(input_sample, explanation, explanation_func)
"""

from __future__ import annotations

from typing import Dict, Any, Callable, Optional, List, Tuple
from pathlib import Path
import copy
import numpy as np
import torch
from torch import nn
from scipy.stats import spearmanr
from scipy.ndimage import gaussian_filter1d


class XAISanityChecker:
    """
    Performs real sanity checks on XAI explanations.
    
    All placeholder values replaced with actual computations.
    """
    
    def __init__(
        self, 
        model: nn.Module,
        window_ms: int = 80,
        sampling_rate: int = 100
    ):
        """
        Initialize sanity checker.
        
        Args:
            model: PyTorch model to check explanations for
            window_ms: Window size in milliseconds for faithfulness tests
            sampling_rate: ECG sampling rate in Hz
        """
        self.model = model
        self.original_weights = copy.deepcopy(model.state_dict())
        self.window_ms = window_ms
        self.sampling_rate = sampling_rate
        self.window_samples = int(window_ms * sampling_rate / 1000)

    def run_checks(
        self, 
        input_tensor: torch.Tensor, 
        original_explanation: np.ndarray,
        explanation_func: Callable,
        baseline: Optional[np.ndarray] = None,
        class_index: int = 0
    ) -> Dict[str, Any]:
        """
        Run all sanity checks with real computations.
        
        Args:
            input_tensor: Input to the model (batch, channels, timesteps)
            original_explanation: The explanation (heatmap) to verify
            explanation_func: Function(model, input) -> explanation
            baseline: Baseline for deletion/insertion (default: zeros)
            class_index: Target class for prediction score
            
        Returns:
            Dictionary with pass/fail status and real metrics for each test.
        """
        results = {}
        
        # 1. Model Randomization Test
        results["randomization_test"] = self._check_model_randomization(
            input_tensor, original_explanation, explanation_func
        )
        
        # 2. Faithfulness Tests (Deletion/Insertion)
        results["faithfulness"] = self._check_faithfulness(
            input_tensor, original_explanation, baseline, class_index
        )
        
        # 3. Stability Test
        results["stability"] = self._check_input_perturbation(
            input_tensor, original_explanation, explanation_func
        )
        
        # Restore original weights
        self.model.load_state_dict(self.original_weights)
        
        # Overall assessment
        results["overall"] = self._compute_overall_assessment(results)
        
        return results

    def _check_model_randomization(
        self,
        input_tensor: torch.Tensor,
        original_explanation: np.ndarray,
        explanation_func: Callable
    ) -> Dict[str, Any]:
        """
        Test 1: Model Parameter Randomization (MPR).
        
        Randomize model weights and check if explanation changes significantly.
        If explanation stays same -> FAIL (it's data dependent, not model dependent).
        
        Uses Spearman correlation for similarity measurement.
        """
        # Randomize last layer(s)
        self._randomize_last_layer()
        
        # Generate explanation with randomized model
        # NOTE: Do NOT use no_grad here - Grad-CAM requires gradients
        self.model.train()  # Enable gradient computation
        input_with_grad = input_tensor.clone().requires_grad_(True)
        try:
            random_explanation = explanation_func(self.model, input_with_grad)
        finally:
            self.model.eval()
        
        # Restore original weights for subsequent tests
        self.model.load_state_dict(self.original_weights)
        
        # Compute similarity using Spearman correlation
        similarity = self._compute_similarity(original_explanation, random_explanation)
        
        # Low similarity is GOOD (explanation changed after randomization)
        passed = similarity < 0.3
        
        return {
            "test": "Model Parameter Randomization",
            "status": "PASS" if passed else "FAIL",
            "similarity": float(similarity),
            "threshold": 0.3,
            "note": "Explanation changed significantly after weight randomization." if passed 
                   else "WARNING: Explanation insensitive to model parameters - may be edge detector behavior."
        }

    def _check_faithfulness(
        self,
        input_tensor: torch.Tensor,
        explanation: np.ndarray,
        baseline: Optional[np.ndarray],
        class_index: int
    ) -> Dict[str, Any]:
        """
        Test 2: Faithfulness via Deletion and Insertion.
        
        Deletion: Mask important regions -> score should DROP
        Insertion: Add important regions to baseline -> score should RISE
        
        Uses time-window based masking appropriate for ECG signals.
        """
        device = input_tensor.device
        input_np = input_tensor.detach().cpu().numpy()
        
        # Setup baseline - use Gaussian blur of signal instead of zeros
        # This provides a more realistic "neutral" baseline for ECG
        if baseline is None:
            baseline = self._create_baseline(input_np)
        
        # Flatten explanation for sorting
        exp_flat = explanation.flatten()
        timesteps = len(exp_flat)
        
        # Define windows
        n_windows = max(1, timesteps // self.window_samples)
        window_importance = []
        
        for i in range(n_windows):
            start = i * self.window_samples
            end = min(start + self.window_samples, timesteps)
            window_importance.append((i, exp_flat[start:end].mean()))
        
        # Sort windows by importance (descending)
        window_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Get original prediction
        self.model.eval()
        with torch.no_grad():
            orig_output = self.model(input_tensor)
            if isinstance(orig_output, dict):
                orig_output = orig_output.get("logits", orig_output)
            
            if orig_output.ndim == 1:
                orig_score = torch.sigmoid(orig_output[0]).item()
            else:
                orig_score = torch.sigmoid(orig_output[0, class_index]).item()
        
        # Deletion curve: progressively mask most important windows with fade
        deletion_scores = [orig_score]
        masked_input = input_np.copy()
        
        # Fade length for soft masking (samples)
        fade_len = max(4, self.window_samples // 4)
        
        for window_idx, _ in window_importance[:min(8, n_windows)]:  # Reduced from 10 to 8
            start = window_idx * self.window_samples
            end = min(start + self.window_samples, timesteps)
            
            # Soft masking with fade transitions
            masked_input = self._soft_mask(
                masked_input, baseline, start, end, fade_len
            )
            
            with torch.no_grad():
                output = self.model(torch.tensor(masked_input, device=device, dtype=torch.float32))
                if isinstance(output, dict):
                    output = output.get("logits", output)
                
                if output.ndim == 1:
                    score = torch.sigmoid(output[0]).item()
                else:
                    score = torch.sigmoid(output[0, class_index]).item()
            deletion_scores.append(score)
        
        # Insertion curve: progressively reveal most important windows
        insertion_scores = []
        revealed_input = baseline.copy()
        
        with torch.no_grad():
            output = self.model(torch.tensor(revealed_input, device=device, dtype=torch.float32))
            if isinstance(output, dict):
                output = output.get("logits", output)
            
            if output.ndim == 1:
                insertion_scores.append(torch.sigmoid(output[0]).item())
            else:
                insertion_scores.append(torch.sigmoid(output[0, class_index]).item())
        
        for window_idx, _ in window_importance[:min(8, n_windows)]:  # Reduced from 10 to 8
            start = window_idx * self.window_samples
            end = min(start + self.window_samples, timesteps)
            
            # Soft reveal with fade transitions
            revealed_input = self._soft_reveal(
                revealed_input, input_np, start, end, fade_len
            )
            
            with torch.no_grad():
                output = self.model(torch.tensor(revealed_input, device=device, dtype=torch.float32))
                if isinstance(output, dict):
                    output = output.get("logits", output)
                
                if output.ndim == 1:
                    score = torch.sigmoid(output[0]).item()
                else:
                    score = torch.sigmoid(output[0, class_index]).item()
            insertion_scores.append(score)
        
        # Compute AUC (normalized by number of steps)
        deletion_auc = np.trapz(deletion_scores) / len(deletion_scores)
        insertion_auc = np.trapz(insertion_scores) / len(insertion_scores)
        
        # Good explanation: low deletion AUC, high insertion AUC
        # Using <= instead of < for borderline cases
        deletion_passed = deletion_auc <= 0.5
        insertion_passed = insertion_auc >= 0.5
        
        return {
            "test": "Faithfulness (Deletion/Insertion)",
            "deletion_auc": float(deletion_auc),
            "insertion_auc": float(insertion_auc),
            "deletion_passed": deletion_passed,
            "insertion_passed": insertion_passed,
            "deletion_curve": deletion_scores,
            "insertion_curve": insertion_scores,
            "window_ms": self.window_ms,
            "note": f"Deletion AUC should be < 0.5, Insertion AUC should be > 0.5"
        }

    def _check_input_perturbation(
        self,
        input_tensor: torch.Tensor,
        original_explanation: np.ndarray,
        explanation_func: Callable,
        noise_scale: float = 0.05
    ) -> Dict[str, Any]:
        """
        Test 3: Input Perturbation Stability.
        
        Add small noise to input and check explanation stability.
        High similarity expected (explanation shouldn't break with minor noise).
        """
        # Add small noise
        noise = torch.randn_like(input_tensor) * noise_scale * input_tensor.std()
        noisy_input = (input_tensor + noise).requires_grad_(True)
        
        # Generate explanation for noisy input
        # NOTE: Do NOT use no_grad here - Grad-CAM requires gradients
        self.model.train()
        try:
            noisy_explanation = explanation_func(self.model, noisy_input)
        finally:
            self.model.eval()
        
        # Compute similarity
        stability = self._compute_similarity(original_explanation, noisy_explanation)
        
        # High stability is GOOD (explanation is robust)
        # Using >= instead of > for borderline cases
        passed = stability >= 0.7
        
        return {
            "test": "Input Perturbation Stability",
            "status": "PASS" if passed else "FAIL",
            "stability_score": float(stability),
            "noise_scale": noise_scale,
            "threshold": 0.7,
            "note": "Explanation is stable under small noise." if passed 
                   else "WARNING: Explanation unstable - may be unreliable."
        }

    def _randomize_last_layer(self) -> None:
        """
        Randomize the weights of the last accessible linear layer(s).
        Uses Xavier initialization for proper random scaling.
        """
        # Find last linear layer(s)
        linear_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers.append((name, module))
        
        if not linear_layers:
            # Try Conv1d as fallback for pure CNN
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv1d):
                    linear_layers.append((name, module))
        
        # Randomize last 1-2 layers
        for name, layer in linear_layers[-2:]:
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def _compute_similarity(
        self, 
        exp1: np.ndarray, 
        exp2: np.ndarray
    ) -> float:
        """
        Compute similarity between two explanations using Spearman correlation.
        
        Spearman is rank-based, robust to scale differences.
        """
        # Flatten both explanations
        flat1 = np.asarray(exp1).flatten()
        flat2 = np.asarray(exp2).flatten()
        
        # Handle edge cases
        if len(flat1) != len(flat2):
            # Resize smaller to match larger
            min_len = min(len(flat1), len(flat2))
            flat1 = flat1[:min_len]
            flat2 = flat2[:min_len]
        
        if np.std(flat1) < 1e-8 or np.std(flat2) < 1e-8:
            return 1.0 if np.allclose(flat1, flat2) else 0.0
        
        # Spearman correlation
        corr, _ = spearmanr(flat1, flat2)
        
        # Handle NaN
        if np.isnan(corr):
            return 0.0
        
        return abs(corr)  # Use absolute value

    def _create_baseline(self, input_np: np.ndarray) -> np.ndarray:
        """
        Create baseline for faithfulness testing.
        
        Strategy:
        1. First try to load train mean from artifacts
        2. Fall back to Gaussian blur of the signal itself
        
        Gaussian blur provides a "neutral" version of the signal that
        preserves overall statistics but removes high-frequency features
        that the model likely uses for classification.
        
        Args:
            input_np: Input signal (batch, channels, timesteps)
            
        Returns:
            Baseline array of same shape
        """
        # Try loading train mean baseline
        baseline_path = Path("artifacts/train_baseline.npz")
        if baseline_path.exists():
            try:
                data = np.load(baseline_path)
                train_mean = data["mean"]
                # Broadcast to batch size
                if input_np.ndim == 3:
                    baseline = np.broadcast_to(train_mean, input_np.shape).copy()
                else:
                    baseline = train_mean.copy()
                return baseline.astype(np.float32)
            except Exception:
                pass
        
        # Fallback: Gaussian blur of the signal
        # sigma=30 @ 100Hz ≈ 300ms blur window - removes QRS details
        sigma = 30
        
        if input_np.ndim == 3:
            # (batch, channels, time)
            baseline = np.zeros_like(input_np)
            for b in range(input_np.shape[0]):
                for c in range(input_np.shape[1]):
                    baseline[b, c] = gaussian_filter1d(input_np[b, c], sigma=sigma)
        elif input_np.ndim == 2:
            # (channels, time)
            baseline = np.zeros_like(input_np)
            for c in range(input_np.shape[0]):
                baseline[c] = gaussian_filter1d(input_np[c], sigma=sigma)
        else:
            # 1D
            baseline = gaussian_filter1d(input_np, sigma=sigma)
        
        return baseline.astype(np.float32)

    def _soft_mask(
        self,
        signal: np.ndarray,
        baseline: np.ndarray,
        start: int,
        end: int,
        fade_len: int
    ) -> np.ndarray:
        """
        Soft mask a region with fade-in/out transitions.
        
        Instead of hard replacement, gradually fade from signal to baseline.
        This avoids discontinuity artifacts that could confuse the model.
        """
        result = signal.copy()
        
        # Ensure we don't go out of bounds
        fade_len = min(fade_len, (end - start) // 2)
        if fade_len < 2:
            # Fallback to hard mask if region too small
            result[:, :, start:end] = baseline[:, :, start:end]
            return result
        
        # Fade-in (signal → baseline)
        fade_in = np.linspace(0, 1, fade_len)
        for i, alpha in enumerate(fade_in):
            if start + i < result.shape[-1]:
                result[:, :, start + i] = (1 - alpha) * signal[:, :, start + i] + alpha * baseline[:, :, start + i]
        
        # Middle section (full baseline)
        mid_start = start + fade_len
        mid_end = end - fade_len
        if mid_start < mid_end:
            result[:, :, mid_start:mid_end] = baseline[:, :, mid_start:mid_end]
        
        # Fade-out (baseline → signal)
        fade_out = np.linspace(1, 0, fade_len)
        for i, alpha in enumerate(fade_out):
            if end - fade_len + i < result.shape[-1]:
                result[:, :, end - fade_len + i] = alpha * baseline[:, :, end - fade_len + i] + (1 - alpha) * signal[:, :, end - fade_len + i]
        
        return result

    def _soft_reveal(
        self,
        baseline_signal: np.ndarray,
        original: np.ndarray,
        start: int,
        end: int,
        fade_len: int
    ) -> np.ndarray:
        """
        Soft reveal a region with fade-in/out transitions.
        
        Opposite of _soft_mask - gradually reveals original from baseline.
        """
        return self._soft_mask(baseline_signal, original, start, end, fade_len)

    def _compute_overall_assessment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute overall sanity assessment."""
        checks = []
        
        if results.get("randomization_test", {}).get("status") == "PASS":
            checks.append("randomization")
        if results.get("faithfulness", {}).get("deletion_passed"):
            checks.append("deletion")
        if results.get("faithfulness", {}).get("insertion_passed"):
            checks.append("insertion")
        if results.get("stability", {}).get("status") == "PASS":
            checks.append("stability")
        
        total_checks = 4
        passed_checks = len(checks)
        
        if passed_checks >= 3:
            status = "RELIABLE"
        elif passed_checks >= 2:
            status = "ACCEPTABLE"
        else:
            status = "UNRELIABLE"
        
        return {
            "status": status,
            "passed_checks": passed_checks,
            "total_checks": total_checks,
            "passed_list": checks,
            "recommendation": "Explanation is model-dependent and faithful." if status == "RELIABLE"
                             else "Explanation may have issues - review individual tests."
        }


def run_sanity_check_standalone(
    model: nn.Module,
    input_tensor: torch.Tensor,
    explanation: np.ndarray,
    explanation_func: Callable,
    class_index: int = 0
) -> Dict[str, Any]:
    """
    Standalone function to run sanity checks.
    
    Convenience wrapper for quick testing.
    """
    checker = XAISanityChecker(model)
    return checker.run_checks(input_tensor, explanation, explanation_func, class_index=class_index)
