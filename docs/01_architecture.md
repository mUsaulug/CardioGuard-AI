# Phase 1: Architecture & Data Flow

**Generated Date:** 2026-01-31
**Methodology:** C4 Model & UML Sequence Analysis based on Source Code.

## 1. C4 System Context

```mermaid
C4Context
    title System Context Diagram for CardioGuard-AI

    Person(clinician, "Clinician", "Uploads ECGs and reviews AI diagnosis.")
    System(cardioguard, "CardioGuard-AI", "Analyzes 12-lead ECGs to detect MI and localize injury.")
    
    Rel(clinician, cardioguard, "Uploads ECG / Views Report", "HTTPS/JSON")
```

## 2. Container Diagram

```mermaid
C4Container
    title Container Diagram for CardioGuard-AI

    Person(clinician, "Clinician", "User")

    Container_Boundary(c1, "CardioGuard-AI System") {
        Container(frontend, "Frontend SPA", "React, Vite", "User Interface for upload and visualization.")
        Container(backend, "Backend Gateway", "FastAPI, Python", "REST API, Validation, Static File Serving.")
        Container(pipeline, "Inference Engine", "PyTorch, XGBoost", "Orchestrates preprocessing, model inference, and XAI.")
        ContainerDb(filesystem, "File System", "Disk", "Stores checkpoints, logs (XGB), and generated XAI artifacts.")
    }

    Rel(clinician, frontend, "Uses", "Browser")
    Rel(frontend, backend, "API Calls", "JSON/HTTP")
    Rel(backend, pipeline, "Invokes", "Python Import")
    Rel(pipeline, filesystem, "Reads Models / Writes Artifacts", "IO")
    Rel(backend, filesystem, "Reads Manifest/Artifacts", "IO")
```

## 3. Component Diagram (Backend & Inference)

```mermaid
classDiagram
    class Backend_Main {
        +POST /predict/superclass
        +GET /runs/{id}/{path}
        -startup_event()
    }
    
    class Inference_Orchestrator {
        +predict(signal)
        -load_models()
    }
    
    class Models {
        +ECGCNN (PyTorch)
        +XGBoost (OVR Ensembles)
    }
    
    class XAI_Engine {
        +UnifiedExplainer
        +ConsistencyGuard
    }

    Backend_Main --> Inference_Orchestrator : Calls predict()
    Inference_Orchestrator --> Models : Runs Inference
    Inference_Orchestrator --> XAI_Engine : Generates Explanation
```

## 4. Sequence Diagram: Prediction Flow

The following sequence describes the `POST /predict/superclass` flow with `explain=True`.

```mermaid
sequenceDiagram
    participant U as User/Frontend
    participant API as FastAPI Backend
    participant P as Pipeline (Orchestrator)
    participant M as Models (CNN/XGB)
    participant X as XAI Engine
    participant FS as FileSystem

    U->>API: POST /predict/superclass (ECG File, explain=True)
    activate API
    API->>API: Parse & Validate Input
    API->>P: predict(signal, models, explain=True)
    activate P
    
    P->>P: Preprocess (Channel First)
    
    par Inference
        P->>M: CNN Forward Pass
        P->>M: XGBoost OVR (on Embeddings)
    end
    
    P->>P: Ensemble & Apply Thresholds
    P->>P: Determine Primary Label
    
    alt MI Detected
        P->>M: Run Localization Model
    end
    
    opt explain=True
        P->>X: Generate Grad-CAM & SHAP
        P->>X: Synthesize Unified Report
        P->>FS: Write manifest.json & Artifacts
    end
    
    P-->>API: Return Result Dict (Result + XAI Info)
    deactivate P
    
    API-->>U: JSON Response (w/ Artifact URLs)
    deactivate API
    
    Note over U, API: Later...
    U->>API: GET /runs/{id}/visuals/plot.png
    API->>FS: Read Artifact (Secure)
    FS-->>API: File Content
    API-->>U: Image
```

## 5. Findings & Evidence
- **Strict Decoupling:** The Sequence Diagram highlights that the Backend Logic (API) never touches the Models directly for inference logic; it delegates entirely to `src.pipeline`.
- **Artifact Serving Pattern:** The API (`serve_xai_artifact`) expects the Pipeline to have *already* written files to disk. It reads `run_id` and paths blindly (but securely), confirming the "Manifest-based" separation of concerns.
- **Evidence:** `src/backend/main.py` L456 calls `pipeline_predict`.
