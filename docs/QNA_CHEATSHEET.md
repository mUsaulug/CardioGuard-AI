# Q&A Cheatsheet

## Architecture
**Q: Why separate Backend and Pipeline?**
A: To decouple serving logic (FastAPI) from ML complexity. It allows strict testing of inference logic without needing a running server, and enables "Fail-Closed" model loading on startup.

**Q: Does the frontend duplicate logic?**
A: No. The frontend is purely a visualization layer. All thresholds and decision rules reside in the backend/pipeline.

## Inference
**Q: How is "NORM" calculated?**
A: `1.0 - max(Pathology_Probs)`. It's a derived residual class, not a direct model output.

**Q: What happens if models disagree?**
A: In the code (`consistency_guard.py`), there is logic for this (Type 1/2 Disagreement). **However**, currently `run_inference_superclass.py` does not call this logic. This is a known audit finding.

## XAI
**Q: Why hardcode `features[-3]`?**
A: It's technical debt. `features[-3]` targets the last meaningful convolutional block in the EfficientNet backbone used. It should be refactored to a named method.

**Q: Is XAI generated on the fly?**
A: Yes, but artifacts are cached. The `explain=true` flag triggers generation, which writes to disk. Subsequent fetches just read the file.

## Security
**Q: Can I read `/etc/passwd` via the artifact endpoint?**
A: No. `serve_xai_artifact` uses `path.resolve().relative_to(RUNS_DIR)`. Any path attempting to go up (`..`) triggers a 400 Error.
