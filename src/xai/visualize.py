"""
XAI Visualization Module.

Utilities to plot ECG signals with Grad-CAM overlays and explainability annotations.

Usage:
    plot_explanation(signal, gradcam_map, output_path, title)
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def plot_12lead_gradcam(
    signal: np.ndarray,
    gradcam_maps: Dict[str, np.ndarray],
    output_path: Path,
    title: str = "ECG Explanation",
    sampling_rate: int = 100
):
    """
    Plot 12-lead ECG with Grad-CAM heatmap overlay for multiple classes.
    
    Args:
        signal: (12, 1000) ECG signal
        gradcam_maps: Dict of {class_name: heatmap_array (1000,)}
        output_path: Path to save the image
        title: Chart title
    """
    
    # Setup plot layout (6 rows x 2 cols standard or 12 rows 1 col)
    # Let's do 12 rows for clarity in MVP
    fig, axes = plt.subplots(12, 1, figsize=(15, 20), sharex=True)
    
    # Create time axis
    timesteps = signal.shape[1]
    time = np.arange(timesteps) / sampling_rate
    
    # Determine colors for classes
    colors = ['r', 'g', 'b', 'purple']
    class_names = list(gradcam_maps.keys())
    
    for i, ax in enumerate(axes):
        # Plot signal
        ax.plot(time, signal[i], 'k', linewidth=0.8, alpha=0.8)
        ax.set_ylabel(LEAD_NAMES[i], rotation=0, labelpad=20, fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Overlay Heatmaps
        # We fill background or overlay line color based on activation
        for idx, cls in enumerate(class_names):
            heatmap = gradcam_maps[cls]
            if heatmap.ndim > 1:
                heatmap = heatmap.squeeze()
            
            # Normalize for visualization if not already
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() + 1e-8)
            
            # Mask low values to avoid clutter
            mask = heatmap > 0.2
            if mask.any():
                # Overlay distinct color for each class
                # Using fill_between for emphasis
                ax.fill_between(
                    time, 
                    signal[i].min(), 
                    signal[i].max(), 
                    where=mask, 
                    color=colors[idx % len(colors)], 
                    alpha=0.3, 
                    label=cls if i == 0 else ""
                )

    axes[-1].set_xlabel("Time (s)", fontsize=14)
    if class_names:
        fig.legend(loc='upper right', fontsize=12, bbox_to_anchor=(0.95, 0.95))
        
    plt.suptitle(title, fontsize=16, y=0.92)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"Explanation plot saved to {output_path}")

def plot_ecg_with_localization(
    signal: np.ndarray,
    localization_probs: Dict[str, float],
    output_path: Path,
    title: str = "MI Localization",
    sampling_rate: int = 100
):
    """
    Plot 12-lead ECG with visual markers for predicted MI regions (mockup).
    Ideally, this would highlight leads relevant to the region, but for now
    it plots the signal and adds a text box with localization probabilities.
    """
    fig, axes = plt.subplots(12, 1, figsize=(15, 20), sharex=True)
    timesteps = signal.shape[1]
    time = np.arange(timesteps) / sampling_rate
    
    for i, ax in enumerate(axes):
        ax.plot(time, signal[i], 'k', linewidth=0.8)
        ax.set_ylabel(LEAD_NAMES[i], rotation=0, labelpad=20, fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.3)
        
    axes[-1].set_xlabel("Time (s)", fontsize=14)
    
    # Add text box with probabilities
    # Filter to only include numeric values (skips "predicted_regions" list)
    text_str = "\n".join([f"{loc}: {prob:.2f}" for loc, prob in localization_probs.items() if isinstance(prob, (int, float))])
    fig.text(0.75, 0.85, text_str, fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.suptitle(title, fontsize=16, y=0.92)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

def plot_ecg_with_prediction(
    signal: np.ndarray,
    prediction = None,
    output_path: Path = None,
    title: str = "ECG Prediction",
    true_label = None,
    save_path: Path = None
):
    """
    Plot 12-lead ECG with prediction results.
    
    Compatible with test signatures.
    """
    # Handle save_path alias
    if save_path is not None:
        output_path = save_path
    
    # Handle prediction as dict or float
    if isinstance(prediction, (int, float)):
        prediction = {"Prediction": prediction}
    if prediction is None:
        prediction = {}
    
    if true_label is not None:
        prediction["True Label"] = true_label
    
    # Simply reuse localization plot structure for now as it's generic enough
    plot_ecg_with_localization(signal, prediction, output_path, title)
    
    return plt.gcf() if hasattr(plt, 'gcf') else None

def plot_gradcam_heatmap(
    heatmap: np.ndarray,
    output_path: Path,
    title: str = "Grad-CAM Heatmap"
):
    """
    Plot a single 1D heatmap.
    """
    if heatmap.ndim > 1:
        heatmap = heatmap.squeeze()
        
    plt.figure(figsize=(10, 2))
    plt.imshow(heatmap[np.newaxis, :], aspect="auto", cmap="jet")
    plt.colorbar()
    plt.title(title)
    plt.yticks([])
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def plot_lead_attention(
    attention_scores: np.ndarray, # (12,)
    output_path: Path,
    title: str = "Lead Attention",
    signal: np.ndarray = None,  # Added for compatibility
    save_path: Path = None  # Alias for output_path
):
    """
    Bar chart of lead importance.
    """
    if save_path is not None:
        output_path = save_path
    
    plt.figure(figsize=(10, 6))
    plt.bar(LEAD_NAMES, attention_scores, color='skyblue')
    plt.title(title)
    plt.ylabel("Relevance Score")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    return plt.gcf() if hasattr(plt, 'gcf') else None


def plot_gradcam_heatmap(
    signal: np.ndarray,
    cam: np.ndarray,
    save_path: Path = None,
    title: str = "Grad-CAM Heatmap",
    sampling_rate: int = 100
):
    """
    Plot ECG signal with Grad-CAM heatmap overlay.
    
    Compatible signature for tests.
    """
    # Ensure proper shapes
    if signal.ndim == 2:
        if signal.shape[0] > signal.shape[1]:
            signal = signal.T  # (C, T) format
    
    cam_flat = cam.flatten() if cam.ndim > 1 else cam
    timesteps = min(signal.shape[1] if signal.ndim == 2 else len(signal), len(cam_flat))
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), height_ratios=[3, 1])
    
    # Top: ECG (first lead or mean)
    if signal.ndim == 2:
        ecg = signal[0, :timesteps]
    else:
        ecg = signal[:timesteps]
    
    time = np.arange(timesteps) / sampling_rate
    axes[0].plot(time, ecg, 'k', linewidth=0.8)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)
    
    # Overlay heatmap as fill
    cam_norm = (cam_flat[:timesteps] - cam_flat[:timesteps].min()) / (cam_flat[:timesteps].max() + 1e-8)
    axes[0].fill_between(time, ecg.min(), ecg, where=cam_norm > 0.3, 
                          color='red', alpha=0.3)
    
    # Bottom: CAM as image
    axes[1].imshow(cam_flat[:timesteps].reshape(1, -1), aspect='auto', cmap='jet', 
                   extent=[0, timesteps/sampling_rate, 0, 1])
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("CAM")
    axes[1].set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def generate_xai_report_png(
    signal: np.ndarray,
    combined_heatmap: np.ndarray,
    shap_features: List[Dict],
    sanity_metrics: Dict,
    prediction: Dict,
    output_path: Path,
    gradcam_heatmap: np.ndarray = None,
    sampling_rate: int = 100
):
    """
    Generate comprehensive 3-panel XAI report PNG.
    
    Panel A: ECG signal with combined heatmap overlay
    Panel B: Top SHAP features bar chart (signed)
    Panel C: Sanity metrics summary box
    
    Args:
        signal: ECG signal (C, T) or (T, C)
        combined_heatmap: Combined SHAP-weighted CAM (T,)
        shap_features: List of top SHAP features with direction/importance
        sanity_metrics: Sanity check results dict
        prediction: Prediction dict with pred_class, pred_proba
        output_path: Path to save PNG
        gradcam_heatmap: Optional raw Grad-CAM for secondary overlay
        sampling_rate: ECG sampling rate
    """
    # Ensure signal shape
    if signal.ndim == 2:
        if signal.shape[0] > signal.shape[1]:
            signal = signal.T
    
    timesteps = signal.shape[1] if signal.ndim == 2 else len(signal)
    time = np.arange(timesteps) / sampling_rate
    
    # Create figure with 3 panels
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1.5, 1], width_ratios=[3, 1])
    
    # ===== Panel A: ECG with heatmap (spans full width) =====
    ax_ecg = fig.add_subplot(gs[0, :])
    
    # Plot first 4 leads stacked
    n_leads = min(4, signal.shape[0] if signal.ndim == 2 else 1)
    lead_offset = 0
    for i in range(n_leads):
        if signal.ndim == 2:
            ecg = signal[i, :]
        else:
            ecg = signal
        ecg_norm = (ecg - ecg.mean()) / (ecg.std() + 1e-8)
        ax_ecg.plot(time, ecg_norm + lead_offset, 'k', linewidth=0.6, alpha=0.8)
        ax_ecg.text(-0.3, lead_offset, LEAD_NAMES[i], fontsize=10, va='center')
        lead_offset += 3
    
    # Overlay combined heatmap
    if combined_heatmap is not None and isinstance(combined_heatmap, np.ndarray) and combined_heatmap.size > 0:
        cam = combined_heatmap.flatten()[:timesteps]
        cam_norm = (cam - cam.min()) / (cam.max() + 1e-8)
        
        # Show as colored background
        for i in range(len(cam_norm) - 1):
            if cam_norm[i] > 0.3:
                ax_ecg.axvspan(time[i], time[i+1], 
                              alpha=cam_norm[i] * 0.4, 
                              color='red', linewidth=0)
    
    ax_ecg.set_xlim(0, time[-1])
    ax_ecg.set_xlabel("Time (s)", fontsize=12)
    ax_ecg.set_ylabel("Leads", fontsize=12)
    ax_ecg.set_title(
        f"Prediction: {prediction.get('pred_class', 'N/A')} "
        f"({prediction.get('pred_proba', 0):.1%})", 
        fontsize=14, fontweight='bold'
    )
    ax_ecg.grid(True, alpha=0.3)
    
    # ===== Panel B: SHAP features bar =====
    ax_shap = fig.add_subplot(gs[1, 0])
    
    if shap_features:
        n_features = min(8, len(shap_features))
        names = [f"Emb-{f['feature_idx']}" for f in shap_features[:n_features]]
        values = [f.get('shap_value', f.get('abs_importance', 0)) for f in shap_features[:n_features]]
        colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]
        
        bars = ax_shap.barh(names, values, color=colors)
        ax_shap.axvline(0, color='black', linewidth=0.5)
        ax_shap.set_xlabel("SHAP Value", fontsize=11)
        ax_shap.set_title("Top Contributing Features", fontsize=12)
        ax_shap.invert_yaxis()
    else:
        ax_shap.text(0.5, 0.5, "No SHAP data", ha='center', va='center', fontsize=12)
        ax_shap.set_axis_off()
    
    # ===== Panel C: Sanity metrics box =====
    ax_sanity = fig.add_subplot(gs[1, 1])
    ax_sanity.set_axis_off()
    
    overall = sanity_metrics.get("overall", {})
    status = overall.get("status", "UNKNOWN")
    passed = overall.get("passed_checks", 0)
    total = overall.get("total_checks", 4)
    
    # Color based on status
    if status == "RELIABLE":
        bg_color = '#d4edda'
        text_color = '#155724'
    elif status == "ACCEPTABLE":
        bg_color = '#fff3cd'
        text_color = '#856404'
    else:
        bg_color = '#f8d7da'
        text_color = '#721c24'
    
    ax_sanity.add_patch(plt.Rectangle((0.05, 0.1), 0.9, 0.8, 
                                       facecolor=bg_color, edgecolor='gray'))
    
    # Sanity text
    sanity_text = f"XAI Quality: {status}\n\nPassed: {passed}/{total} checks"
    
    # Add individual metrics
    rand_sim = sanity_metrics.get("randomization_test", {}).get("similarity", "N/A")
    del_auc = sanity_metrics.get("faithfulness", {}).get("deletion_auc", "N/A")
    ins_auc = sanity_metrics.get("faithfulness", {}).get("insertion_auc", "N/A")
    stab = sanity_metrics.get("stability", {}).get("stability_score", "N/A")
    
    if isinstance(rand_sim, float):
        sanity_text += f"\n\nRand Sim: {rand_sim:.2f} (<0.3)"
    if isinstance(del_auc, float):
        sanity_text += f"\nDel AUC: {del_auc:.2f} (<0.5)"
    if isinstance(ins_auc, float):
        sanity_text += f"\nIns AUC: {ins_auc:.2f} (>0.5)"
    if isinstance(stab, float):
        sanity_text += f"\nStability: {stab:.2f} (>0.7)"
    
    ax_sanity.text(0.5, 0.5, sanity_text, ha='center', va='center', 
                   fontsize=10, color=text_color,
                   transform=ax_sanity.transAxes)
    ax_sanity.set_title("Sanity Checks", fontsize=12)
    
    # ===== Panel D: Heatmap strip =====
    ax_heat = fig.add_subplot(gs[2, :])
    
    if combined_heatmap is not None and isinstance(combined_heatmap, np.ndarray) and combined_heatmap.size > 0:
        cam = combined_heatmap.flatten()[:timesteps]
        ax_heat.imshow(cam.reshape(1, -1), aspect='auto', cmap='hot',
                       extent=[0, time[-1], 0, 1])
        ax_heat.set_xlabel("Time (s)", fontsize=11)
        ax_heat.set_ylabel("Saliency", fontsize=11)
        ax_heat.set_yticks([])
    else:
        ax_heat.text(0.5, 0.5, "No heatmap data", ha='center', va='center')
        ax_heat.set_axis_off()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return output_path
