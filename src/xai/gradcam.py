"""Grad-CAM implementation for CNN models."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
from torch import nn


class GradCAM:
    """Compute Grad-CAM heatmaps for a target layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_, __, output):
            self.activations = output

        def backward_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, inputs: torch.Tensor, class_index: int | None = None) -> np.ndarray:
        """Generate Grad-CAM heatmap for the given inputs."""

        self.model.zero_grad(set_to_none=True)
        output = self.model(inputs)
        if isinstance(output, dict):
            logits = output.get("logits")
            if logits is None:
                raise KeyError("GradCAM expects 'logits' in model output dict.")
        else:
            logits = output
        if isinstance(logits, (tuple, list)):
            raise TypeError("GradCAM expects model output to be logits only.")
        if logits.dim() == 1:
            score = logits.sum()
        elif logits.dim() == 2:
            if class_index is None:
                class_index = int(torch.argmax(logits, dim=1)[0])
            score = logits[:, class_index].sum()
        else:
            raise ValueError("Logits tensor must be 1D or 2D.")
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations are not captured.")

        if self.gradients.dim() == 3:
            weights = torch.mean(self.gradients, dim=2, keepdim=True)
        elif self.gradients.dim() == 4:
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        else:
            raise ValueError("Gradients tensor must be 3D or 4D.")
        cam = torch.sum(weights * self.activations, dim=1)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()


# Class index mapping for multi-label superclass
SUPERCLASS_CLASS_IDX = {"MI": 0, "STTC": 1, "CD": 2, "HYP": 3}


def generate_relevant_gradcam(
    model: nn.Module,
    target_layer: nn.Module,
    inputs: torch.Tensor,
    probs: dict,
    thresholds: dict,
    top_k: int = 3,
) -> dict:
    """
    Generate GradCAM only for relevant classes (threshold exceeded).
    
    Classes are sorted by probability (highest first) and limited to top_k.
    
    Args:
        model: CNN model
        target_layer: Target layer for GradCAM
        inputs: Input tensor (batch, channels, timesteps)
        probs: Dict of class -> probability
        thresholds: Dict of class -> threshold
        top_k: Maximum number of classes to generate CAM for
        
    Returns:
        Dict of class_name -> cam_array
    """
    # Find relevant classes (above threshold)
    relevant = [
        cls for cls in ["MI", "STTC", "CD", "HYP"]
        if probs.get(cls, 0) >= thresholds.get(cls, 0.5)
    ]
    
    # Sort by probability (highest first) and limit to top_k
    relevant = sorted(relevant, key=lambda c: probs.get(c, 0), reverse=True)[:top_k]
    
    if not relevant:
        return {}
    
    gradcam = GradCAM(model, target_layer)
    results = {}
    
    for cls in relevant:
        class_idx = SUPERCLASS_CLASS_IDX.get(cls)
        if class_idx is not None:
            cam = gradcam.generate(inputs, class_index=class_idx)
            results[cls] = cam
    
    return results
