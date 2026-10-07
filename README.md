# Video Analysis Experiments

This repository contains five related stages for video understanding experiments. Each stage is self-contained and has its own README, setup script, and Python dependencies.

| Directory | Purpose |
| --- | --- |
| `00_prototype/` | Qwen3-VL event extraction prototype |
| `01_low_freq_event/` | Low-frequency event detection |
| `02_shot_boundary_detection/` | Shot boundary detection |
| `03_adaptive_sampling/` | Motion-aware adaptive frame sampling |
| `04_entity_extraction/` | Entity tracking, identity linking, and story summaries |

## Start after cloning

Choose a stage and follow its README. For example:

```bash
cd 03_adaptive_sampling
./setup.sh
./run.sh --help
```

The setup scripts create stage-local virtual environments. They may download Python packages and model weights. Large downloaded media and generated output directories are kept out of Git; small intermediate inputs and fixtures that help resume or inspect a stage may be tracked. Results are written under each stage's `outputs/` directory.

Large local-only data such as `.venv/`, `.cache/`, `outputs/`, downloaded media, and model weights is excluded by `.gitignore`.
