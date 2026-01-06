"""
XAI Batch Test Script

1. Creates test_samples directory
2. Generates multiple NPZ files with varied ECG patterns
3. Runs XAI report generation on all samples
4. Prints summary of results

Usage:
    python scripts/run_xai_batch_test.py
"""

import numpy as np
from pathlib import Path
import subprocess
import sys
import json


def generate_synthetic_ecg(sample_id: int, pattern: str = "normal") -> np.ndarray:
    """
    Generate synthetic 12-lead ECG signal.
    
    Args:
        sample_id: Unique ID for reproducibility
        pattern: 'normal', 'mi', 'sttc', 'cd', 'hyp'
    
    Returns:
        ECG signal (12, 1000) @ 100Hz = 10 seconds
    """
    np.random.seed(sample_id)
    
    # Base signal: 12 leads, 1000 timesteps
    t = np.linspace(0, 10, 1000)
    ecg = np.zeros((12, 1000))
    
    # Generate base rhythm (simplified QRS complex pattern)
    for lead in range(12):
        # Heart rate ~70 bpm = ~1.17 Hz
        hr_variation = 0.95 + 0.1 * np.random.rand()
        freq = 1.17 * hr_variation
        
        # P-wave, QRS, T-wave simplified simulation
        for beat in range(12):  # ~12 beats in 10 seconds
            beat_center = beat / freq + 0.1 * np.random.randn()
            if beat_center < 0 or beat_center > 10:
                continue
            
            # QRS (sharp spike)
            qrs_width = 0.08 + 0.02 * np.random.rand()
            qrs_amp = (0.8 + 0.4 * np.random.rand()) * (1 if lead < 6 else 0.6)
            qrs = qrs_amp * np.exp(-((t - beat_center) ** 2) / (2 * qrs_width ** 2))
            
            # T-wave (broader)
            t_offset = 0.25 + 0.05 * np.random.rand()
            t_width = 0.15 + 0.03 * np.random.rand()
            t_amp = 0.3 + 0.2 * np.random.rand()
            t_wave = t_amp * np.exp(-((t - beat_center - t_offset) ** 2) / (2 * t_width ** 2))
            
            ecg[lead] += qrs + t_wave
    
    # Add pattern-specific modifications
    if pattern == "mi":
        # ST elevation in leads V1-V4 (indices 6-9)
        for lead in range(6, 10):
            ecg[lead] += 0.3 * np.sin(2 * np.pi * 0.5 * t) + 0.2
        # Q waves
        ecg[0] -= 0.15 * np.exp(-((t - 0.5) ** 2) / 0.01)
        
    elif pattern == "sttc":
        # ST-T changes: inverted T waves
        for lead in [0, 1, 5, 6]:
            ecg[lead] -= 0.25 * np.sin(2 * np.pi * 1.2 * t)
            
    elif pattern == "cd":
        # Conduction delay: widened QRS
        ecg = np.roll(ecg, 10, axis=1)
        ecg[:, :10] = 0
        # Add irregular beats
        ecg[:, 300:320] += 0.5 * np.random.randn(12, 20)
        
    elif pattern == "hyp":
        # Hypertrophy: increased voltage
        ecg *= 1.5
        # Deep S waves in V1-V2
        ecg[6:8] -= 0.4
    
    # Add noise
    ecg += 0.05 * np.random.randn(12, 1000)
    
    return ecg.astype(np.float32)


def create_test_samples(output_dir: Path, n_samples: int = 10):
    """Create test sample NPZ files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    patterns = ["normal", "mi", "sttc", "cd", "hyp"]
    created = []
    
    for i in range(n_samples):
        # Cycle through patterns
        pattern = patterns[i % len(patterns)]
        sample_id = f"sample_{i:03d}_{pattern}"
        
        ecg = generate_synthetic_ecg(i, pattern)
        
        # Save as NPZ
        filepath = output_dir / f"{sample_id}.npz"
        np.savez(filepath, signal=ecg, label=pattern)
        
        created.append({
            "id": sample_id,
            "pattern": pattern,
            "path": str(filepath)
        })
        print(f"  Created: {sample_id}.npz ({pattern})")
    
    return created


def run_xai_report(input_dir: Path, skip_sanity: bool = True):
    """Run XAI report generation."""
    cmd = [
        sys.executable, "-m", "src.pipeline.generate_xai_report",
        "--input-dir", str(input_dir),
        "--task", "multiclass"
    ]
    
    if skip_sanity:
        cmd.append("--skip-sanity")
    
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def main():
    print("=" * 60)
    print("XAI Batch Test Pipeline")
    print("=" * 60)
    
    # Configuration
    test_dir = Path("reports/xai/test_samples")
    n_samples = 10  # Create 10 test samples
    
    # Step 1: Create test samples
    print(f"\n[1/3] Creating {n_samples} test samples...")
    samples = create_test_samples(test_dir, n_samples)
    print(f"  ✓ Created {len(samples)} samples in {test_dir}")
    
    # Step 2: Run XAI report generation (fast mode first)
    print(f"\n[2/3] Running XAI report generation (skip-sanity mode)...")
    exit_code = run_xai_report(test_dir, skip_sanity=True)
    
    if exit_code != 0:
        print(f"  ✗ XAI report generation failed with exit code {exit_code}")
        return
    
    print(f"  ✓ XAI reports generated successfully")
    
    # Step 3: Summary
    print(f"\n[3/3] Summary")
    print("-" * 40)
    
    # Find latest run
    runs_dir = Path("reports/xai/runs")
    if runs_dir.exists():
        runs = sorted(runs_dir.iterdir(), key=lambda x: x.name, reverse=True)
        if runs:
            latest_run = runs[0]
            manifest_path = latest_run / "manifest.json"
            
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                
                print(f"  Run ID: {manifest.get('run_id')}")
                print(f"  Samples: {manifest.get('n_samples')}")
                print(f"  Reliability: {manifest.get('reliability_rate', 0):.1%}")
                print(f"  Duration: {manifest.get('duration_seconds', 0):.2f}s")
                print(f"\n  Output: {latest_run}")
                print(f"  - cards.jsonl")
                print(f"  - visuals/*.png")
                print(f"  - text/*__narrative.md")
    
    print("\n" + "=" * 60)
    print("Done! Check the reports directory for outputs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
