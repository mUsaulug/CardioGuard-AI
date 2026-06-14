"""
CardioGuard-AI Configuration Module

Central configuration for the CardioGuard-AI project.
All paths, constants, and hyperparameters should be defined here.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import json


# Minimum likelihood for PTB-XL label assignment (0–100 scale). Training scripts use this SSOT.
DEFAULT_MIN_LIKELIHOOD = 0.0


@dataclass
class PTBXLConfig:
    """Configuration for PTB-XL dataset processing."""
    
    # Dataset paths (relative to project root by default)
    data_root: Path = field(
        default_factory=lambda: Path("physionet.org/files/ptb-xl/1.0.3")
    )
    metadata_root: Path = field(default_factory=lambda: Path("data/ptbxl_meta"))
    
    # Sampling rate: 100 or 500 Hz
    sampling_rate: int = 100
    
    # Standard PTB-XL split using strat_fold
    # Train: folds 1-8, Val: fold 9, Test: fold 10
    train_folds: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8])
    val_folds: List[int] = field(default_factory=lambda: [9])
    test_folds: List[int] = field(default_factory=lambda: [10])
    
    # Random seed for reproducibility
    random_seed: int = 42

    # Minimum likelihood threshold for label assignment (PTB-XL 0–100 scale)
    min_likelihood: float = DEFAULT_MIN_LIKELIHOOD

    # Label task type
    task: str = "binary"  # "binary" for MI vs NORM, "multiclass" for 5-class
    
    @property
    def metadata_path(self) -> Path:
        """Path to ptbxl_database.csv"""
        return self.metadata_root / "ptbxl_database.csv"
    
    @property
    def scp_statements_path(self) -> Path:
        """Path to scp_statements.csv"""
        return self.metadata_root / "scp_statements.csv"
    
    @property
    def records_path(self) -> Path:
        """Path to signal records directory"""
        if self.sampling_rate == 100:
            return self.data_root / "records100"
        else:
            return self.data_root / "records500"
    
    @property
    def filename_column(self) -> str:
        """Column name for signal file paths"""
        if self.sampling_rate == 100:
            return "filename_lr"
        else:
            return "filename_hr"


@dataclass
class RAGSource:
    """RAG knowledge source definition."""

    name: str
    description: str
    path: Optional[Path] = None
    notes: Optional[str] = None


# =============================================================================
# Task Schema — single source of truth for all task definitions
# =============================================================================

# Pathology classes predicted by the superclass model (NORM is derived, not predicted)
SUPERCLASS_LABELS = ["MI", "STTC", "CD", "HYP"]

# MI localization anatomical regions (derived from PTB-XL SCP codes via MI_CODE_TO_REGIONS)
MI_LOCALIZATION_LABELS = ["AMI", "ASMI", "ALMI", "IMI", "LMI"]

# Turkish display labels — canonical source for API + frontend conformance
PATHOLOGY_LABELS_TR: Dict[str, str] = {
    "MI": "Miyokard Enfarktüsü",
    "STTC": "ST/T Değişikliği",
    "CD": "İletim Bozukluğu",
    "HYP": "Hipertrofi",
    "NORM": "Normal",
}

MI_LOCALIZATION_LABELS_TR: Dict[str, str] = {
    "AMI": "Anterior MI",
    "ASMI": "Anteroseptal MI",
    "ALMI": "Anterolateral MI",
    "IMI": "İnferior MI",
    "LMI": "Lateral MI",
}

TRIAGE_TR: Dict[str, str] = {
    "HIGH": "YÜKSEK",
    "MEDIUM": "ORTA",
    "LOW": "DÜŞÜK",
    "REVIEW": "İNCELEME",
}

GLOSSARY: Dict[str, str] = {
    "MI": "Miyokard Enfarktüsü — kalp kasında kan akımının kesilmesi sonucu hasar",
    "STTC": "ST/T değişikliği — iskemi veya repolarizasyon bozukluğu bulguları",
    "CD": "İletim bozukluğu — kalp elektrik iletiminde gecikme veya blok",
    "HYP": "Hipertrofi — kalp odacıklarında veya duvarında kalınlaşma",
    "NORM": "Normal EKG — belirgin patoloji tespit edilmedi",
    "AMI": "Anterior miyokard enfarktüsü — V3-V4 derivasyonları",
    "ASMI": "Anteroseptal miyokard enfarktüsü — V1-V4 derivasyonları",
    "ALMI": "Anterolateral miyokard enfarktüsü — V3-V6, I, aVL",
    "IMI": "İnferior miyokard enfarktüsü — II, III, aVF derivasyonları",
    "LMI": "Lateral miyokard enfarktüsü — I, aVL, V5-V6 derivasyonları",
    "Consistency Guard": (
        "İki bağımsız MI modelinin uyumunu kontrol eden güvenlik katmanı"
    ),
}

CLINICAL_DISCLAIMER = "Bu sistem tanı koymaz; klinik karar destek aracıdır."

# Output dimensions per task — must match model checkpoint head sizes
TASK_OUTPUT_DIMS = {
    "binary": 1,
    "superclass": 4,
    "mi_localization": 5,
}

# Labels per task
TASK_LABELS = {
    "binary": ["MI"],
    "superclass": SUPERCLASS_LABELS,
    "mi_localization": MI_LOCALIZATION_LABELS,
}

# Fingerprint of MI_CODE_TO_REGIONS mapping (sha256[:16] of sorted dict)
# Update this if MI_CODE_TO_REGIONS changes in src/data/mi_localization.py
MI_LOCALIZATION_FINGERPRINT = "8ab274e06afa1be8"

# Diagnostic class mappings
DIAGNOSTIC_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
DIAGNOSTIC_PRIORITY = ["MI", "STTC", "CD", "HYP", "NORM"]

# All MI-related SCP codes from scp_statements.csv
# These include both definite MI and subendocardial injury patterns
MI_CODES = [
    "IMI",    # Inferior MI
    "ASMI",   # Anteroseptal MI
    "AMI",    # Anterior MI
    "ALMI",   # Anterolateral MI
    "LMI",    # Lateral MI
    "ILMI",   # Inferolateral MI
    "IPLMI",  # Inferoposterolateral MI
    "IPMI",   # Inferoposterior MI
    "PMI",    # Posterior MI
    # Subendocardial injury codes (also MI class)
    "INJIN",  # Subendocardial injury inferior
    "INJAL",  # Subendocardial injury anterolateral
    "INJAS",  # Subendocardial injury anteroseptal
    "INJIL",  # Subendocardial injury inferolateral
    "INJLA",  # Subendocardial injury lateral
]

# RAG sources (placeholders for future ingestion)
RAG_SOURCES = [
    RAGSource(
        name="MI Guideline PDF",
        description="Myocardial infarction guideline document (to be added).",
        path=None,
        notes="Planned: official MI guideline PDF.",
    ),
    RAGSource(
        name="Literature Notes",
        description="Curated literature notes for MI risk factors (to be added).",
        path=None,
        notes="Planned: internal clinical notes.",
    ),
]


DEFAULT_THRESHOLDS_ARTIFACT = Path("artifacts/thresholds_superclass.json")
DEFAULT_ENSEMBLE_CNN_WEIGHT = 0.15  # fallback; prefer artifact file


def load_superclass_thresholds_artifact(path: Optional[Path] = None) -> dict:
    """Load thresholds JSON artifact (thresholds + ensemble_weight)."""
    artifact_path = path or DEFAULT_THRESHOLDS_ARTIFACT
    with open(artifact_path, encoding="utf-8") as f:
        return json.load(f)


def get_ensemble_cnn_weight(path: Optional[Path] = None) -> float:
    """CNN weight in ensemble — single source from thresholds artifact."""
    try:
        data = load_superclass_thresholds_artifact(path)
        return float(data.get("ensemble_weight", DEFAULT_ENSEMBLE_CNN_WEIGHT))
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        return DEFAULT_ENSEMBLE_CNN_WEIGHT


def get_default_config(project_root: Optional[Path] = None) -> PTBXLConfig:
    """
    Get default configuration with optional project root override.
    
    Args:
        project_root: Optional path to project root. If None, uses current working directory.
        
    Returns:
        PTBXLConfig instance with paths resolved relative to project_root.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    
    return PTBXLConfig(
        data_root=project_root / "physionet.org" / "files" / "ptb-xl" / "1.0.3",
        metadata_root=project_root / "data" / "ptbxl_meta",
    )
