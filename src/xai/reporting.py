"""
XAI Reporting Infrastructure for CardioGuard-AI.

Creates structured output files for XAI explanations:
- manifest.json: Run-level metadata
- cards.jsonl: Sample-level explanation cards (JSON Lines)
- tables/*.parquet: Structured tables for analysis
- tensors/*.npz: Full heatmaps and arrays
- visuals/*.png: Visual reports
- text/*.md: Narrative for RAG

Usage:
    reporter = XAIReporter(run_id, output_dir, task)
    reporter.add_sample(sample_id, explanation, sanity, prediction)
    reporter.finalize()  # Creates manifest
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
import json
import subprocess
import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def get_git_sha() -> str:
    """Get current git short SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def generate_run_id(model_tag: str, task: str) -> str:
    """Generate run ID in standard format."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_sha = get_git_sha()
    return f"{timestamp}__{git_sha}__{model_tag}__{task}"


class XAIReporter:
    """
    Manages XAI report generation for a run.
    
    Creates all output files in structured directory.
    """
    
    def __init__(
        self,
        run_id: str,
        output_dir: Union[str, Path],
        task: str,
        model_id: str = "unknown",
        xgb_id: str = "unknown",
        baseline_source: str = "train_mean"
    ):
        """
        Initialize reporter.
        
        Args:
            run_id: Unique identifier for this run
            output_dir: Base output directory
            task: Task type (binary, multiclass, localization)
            model_id: CNN model identifier
            xgb_id: XGBoost model identifier
            baseline_source: Baseline used for faithfulness tests
        """
        self.run_id = run_id
        self.base_dir = Path(output_dir) / run_id
        self.task = task
        self.model_id = model_id
        self.xgb_id = xgb_id
        self.baseline_source = baseline_source
        
        # Create directory structure
        self._create_directories()
        
        # Storage for aggregation
        self.samples: List[Dict[str, Any]] = []
        self.cards_file = open(self.base_dir / "cards.jsonl", "w", encoding="utf-8")
        
        # Track statistics
        self.start_time = datetime.utcnow()
    
    def _create_directories(self) -> None:
        """Create output directory structure."""
        (self.base_dir / "tables").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "tensors").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "visuals").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "text").mkdir(parents=True, exist_ok=True)
    
    def add_sample(
        self,
        sample_id: str,
        explanation: Dict[str, Any],
        sanity: Dict[str, Any],
        prediction: Dict[str, Any],
        signal: Optional[np.ndarray] = None,
        true_label: Optional[str] = None,
        narrative: Optional[str] = None
    ) -> None:
        """
        Add a sample to the report.
        
        Args:
            sample_id: Unique sample identifier
            explanation: Output from CombinedExplainer
            sanity: Output from XAISanityChecker
            prediction: Prediction dict with pred_class, pred_proba, etc.
            signal: Original ECG signal (optional, for visuals)
            true_label: Ground truth label
            narrative: Pre-generated narrative text
        """
        # Create explanation card
        card = self._create_card(sample_id, explanation, sanity, prediction, true_label)
        
        # Write to JSONL
        self.cards_file.write(json.dumps(card, default=str) + "\n")
        
        # Store for aggregation
        self.samples.append({
            "sample_id": sample_id,
            "pred_class": prediction.get("pred_class"),
            "pred_proba": prediction.get("pred_proba"),
            "true_label": true_label,
            "sanity_status": sanity.get("overall", {}).get("status"),
            "passed_checks": sanity.get("overall", {}).get("passed_checks", 0)
        })
        
        # Save tensors
        self._save_tensors(sample_id, explanation, sanity)
        
        # Save narrative
        if narrative:
            self._save_narrative(sample_id, narrative)
    
    def _create_card(
        self,
        sample_id: str,
        explanation: Dict[str, Any],
        sanity: Dict[str, Any],
        prediction: Dict[str, Any],
        true_label: Optional[str]
    ) -> Dict[str, Any]:
        """Create structured explanation card."""
        return {
            "meta": {
                "run_id": self.run_id,
                "sample_id": sample_id,
                "task": self.task,
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "model_id": self.model_id,
                "xgb_id": self.xgb_id
            },
            "prediction": {
                "pred_class": prediction.get("pred_class"),
                "pred_proba": prediction.get("pred_proba"),
                "runnerup_class": prediction.get("runnerup"),
                "runnerup_proba": prediction.get("runnerup_proba"),
                "true_label": true_label,
                "correct": true_label == prediction.get("pred_class") if true_label else None
            },
            "xai_shap": explanation.get("shap", {}),
            "xai_gradcam": explanation.get("gradcam", {}),
            "xai_combined": explanation.get("combined", {}),
            "contrastive": explanation.get("contrastive", {}),
            "sanity": {
                "overall": sanity.get("overall", {}),
                "randomization": sanity.get("randomization_test", {}),
                "faithfulness": {
                    "deletion_auc": sanity.get("faithfulness", {}).get("deletion_auc"),
                    "insertion_auc": sanity.get("faithfulness", {}).get("insertion_auc")
                },
                "stability": sanity.get("stability", {})
            }
        }
    
    def _save_tensors(
        self,
        sample_id: str,
        explanation: Dict[str, Any],
        sanity: Dict[str, Any]
    ) -> None:
        """Save tensor data as NPZ."""
        tensors = {}
        
        # SHAP values
        shap_data = explanation.get("shap", {})
        if shap_data.get("shap_values") is not None:
            tensors["shap_values"] = np.asarray(shap_data["shap_values"])
        
        # Grad-CAM heatmap
        gradcam_data = explanation.get("gradcam", {})
        if gradcam_data.get("heatmap") is not None:
            tensors["gradcam_heatmap"] = np.asarray(gradcam_data["heatmap"])
        
        # Combined heatmap
        combined_data = explanation.get("combined", {})
        if combined_data.get("heatmap") is not None:
            tensors["combined_heatmap"] = np.asarray(combined_data["heatmap"])
        
        # Faithfulness curves
        faith_data = sanity.get("faithfulness", {})
        if faith_data.get("deletion_curve"):
            tensors["deletion_curve"] = np.array(faith_data["deletion_curve"])
        if faith_data.get("insertion_curve"):
            tensors["insertion_curve"] = np.array(faith_data["insertion_curve"])
        
        if tensors:
            np.savez(
                self.base_dir / "tensors" / f"{sample_id}.npz",
                **tensors
            )
    
    def _save_narrative(self, sample_id: str, narrative: str) -> None:
        """Save narrative markdown."""
        path = self.base_dir / "text" / f"{sample_id}__narrative.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(narrative)
    
    def save_visual_report(
        self,
        sample_id: str,
        figure_path: Path
    ) -> None:
        """Record that a visual report was saved."""
        # Visual is saved by visualize module, we just track it
        pass
    
    def finalize(self) -> Path:
        """
        Finalize report and create manifest.
        
        Returns:
            Path to manifest.json
        """
        self.cards_file.close()
        
        # Create summary tables if pandas available
        summary_file = None
        if PANDAS_AVAILABLE and self.samples:
            summary_file = self._create_summary_tables()
        
        # Create manifest
        manifest = self._create_manifest(summary_file)
        manifest_path = self.base_dir / "manifest.json"
        
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
        
        return manifest_path
    
    def _create_summary_tables(self) -> str:
        """
        Create summary tables.
        
        Returns:
            Filename of the created summary table.
        """
        df = pd.DataFrame(self.samples)
        try:
            filename = "sample_summary.parquet"
            df.to_parquet(self.base_dir / "tables" / filename, index=False)
            return filename
        except ImportError:
            # Fallback to CSV if pyarrow/fastparquet not installed
            print("Warning: Parquet dependencies (pyarrow) missing. Saving summary as CSV.")
            filename = "sample_summary.csv"
            df.to_csv(self.base_dir / "tables" / filename, index=False)
            return filename
        except Exception as e:
            print(f"Warning: Failed to save summary table: {e}")
            return None
    
    def _create_manifest(self, summary_file: Optional[str] = None) -> Dict[str, Any]:
        """Create run manifest."""
        end_time = datetime.utcnow()
        
        # Compute statistics
        n_samples = len(self.samples)
        n_passed = sum(1 for s in self.samples if s.get("sanity_status") == "RELIABLE")
        n_correct = sum(1 for s in self.samples if s.get("correct") is True)
        
        return {
            "run_id": self.run_id,
            "task": self.task,
            "model_id": self.model_id,
            "xgb_id": self.xgb_id,
            "baseline_source": self.baseline_source,
            "git_sha": get_git_sha(),
            "start_time": self.start_time.isoformat() + "Z",
            "end_time": end_time.isoformat() + "Z",
            "duration_seconds": (end_time - self.start_time).total_seconds(),
            "n_samples": n_samples,
            "n_reliable_explanations": n_passed,
            "reliability_rate": n_passed / n_samples if n_samples > 0 else 0,
            "accuracy": n_correct / n_samples if n_samples > 0 else None,
            "files": {
                "cards": "cards.jsonl",
                "tables": f"tables/{summary_file}" if summary_file else None,
                "tensors": "tensors/",
                "visuals": "visuals/",
                "narratives": "text/"
            }
        }


def quick_report(
    sample_id: str,
    explanation: Dict[str, Any],
    sanity: Dict[str, Any],
    prediction: Dict[str, Any],
    output_dir: Union[str, Path],
    task: str = "multiclass"
) -> Path:
    """
    Quick single-sample report generation.
    
    Returns path to created directory.
    """
    run_id = generate_run_id("quick", task)
    reporter = XAIReporter(run_id, output_dir, task)
    reporter.add_sample(sample_id, explanation, sanity, prediction)
    reporter.finalize()
    return reporter.base_dir
