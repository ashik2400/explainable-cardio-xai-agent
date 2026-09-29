# Explainable Agentic Clinical Decision Support System
### Cardiovascular Risk & Diagnosis Prediction (ECG + EHR/Labs)

An explainable, multimodal AI system that predicts cardiovascular diagnoses from ECG signals and structured lab/EHR data, explains its predictions using SHAP, retrieves supporting medical literature from PubMed keyed on the top contributing features, and uses an LLM-based orchestrator to generate a clinician-readable report grounded in that evidence.

Built as a course project for **"Responsible & Safe AI Systems"** (Swayam, 12-week course, 8-week accelerated project timeline).

---

## Table of Contents
- [Project Motivation](#project-motivation)
- [Core Pipeline](#core-pipeline)
- [Novelty / Core Contribution](#novelty--core-contribution)
- [Scope Boundaries](#scope-boundaries-intentional)
- [Tech Stack](#tech-stack)
- [Repo Structure](#repo-structure)
- [Setup](#setup)
- [Data Source & Reproduction](#data-source--reproduction)
- [MLOps: DVC & MLflow](#mlops-dvc--mlflow)
- [Roadmap / Phase Status](#roadmap--phase-status)
- [Key Risks](#key-risks)

---

## Project Motivation

Most existing explainable-AI systems in clinical settings either (a) predict + explain, or (b) retrieve + generate — few explicitly chain **SHAP-attributed features → literature-grounded evidence → generated report** as a single connected pipeline. This project's differentiating contribution is that link, not the individual components (which build on existing pretrained models and libraries rather than novel architectures).

## Core Pipeline

1. **Data** — PTB-XL (12-lead ECG signals + diagnostic labels) + structured lab/EHR-style features (age, sex, height, weight, from the same dataset).
2. **Per-modality models** — a pretrained ECG classifier + a lab/EHR classifier (not trained from scratch).
3. **Fusion** — feature-level concatenation of extracted representations from each modality → a lightweight trained classifier, producing a genuine multimodal prediction rather than two independent pipelines.
4. **Explainability** — SHAP applied to the fused model, producing ranked feature contributions per prediction.
5. **Evidence retrieval** — top SHAP features translated into PubMed search queries via NCBI E-utilities (`esearch` + `efetch`), results filtered for relevance and summarized.
6. **Report generation** — an LLM-based orchestrator (Groq API, tool-using) synthesizes the prediction, SHAP explanation, and retrieved evidence into a clinician-facing report.

## Novelty / Core Contribution

See [Project Motivation](#project-motivation) above — the SHAP → evidence → report chain is the core contribution; individual components rely on existing pretrained models and libraries.

## Scope Boundaries (intentional)

- Two modalities only (EHR/labs + ECG) — imaging (X-ray) explicitly deferred to future work.
- Pretrained models used for per-modality prediction — no diagnostic model trained from scratch.
- Orchestrator is a scripted/tool-using LLM pipeline, not a full autonomous multi-step agentic framework.
- Evidence retrieval is a simple RAG setup over PubMed abstracts — no custom retrieval model training.

## Tech Stack

| Layer | Tool |
|---|---|
| ECG / signal handling | `wfdb`, `neurokit2` |
| Deep learning | `torch` (CUDA, RTX 3050 4GB) |
| Explainability | `shap` |
| Evidence retrieval | `biopython` (NCBI E-utilities) |
| LLM orchestrator | **Groq API** |
| Data versioning | **DVC** |
| Experiment tracking | **MLflow** |
| Environment | **conda** (not venv — needed for CUDA toolkit management alongside Python packages) |

## Repo Structure

```
swayam/
├── data/
│   ├── raw/              # PTB-XL CSVs (DVC-tracked, not in git)
│   └── processed/
│       └── signals/      # reshaped per-record .npy arrays (DVC-tracked)
├── notebooks/            # exploration notebooks
├── src/
│   ├── data/              # loading, cleaning, reshaping scripts
│   ├── models/            # per-modality + fusion model code
│   ├── explainability/    # SHAP pipeline
│   ├── retrieval/         # PubMed E-utilities integration
│   └── report/            # LLM orchestrator (Groq)
├── models/                 # saved model checkpoints (gitignored)
├── reports/                 # generated clinician reports, eval writeups
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```

## Setup

### 1. Environment (conda)

```powershell
conda create -n cardio-xai python=3.10 -y
conda activate cardio-xai

# torch with CUDA support (check nvidia-smi first for your supported CUDA version)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

pip install -r requirements.txt
```

### 2. Environment variables

Copy `.env.example` to `.env` and fill in:
- `GROQ_API_KEY` — for the Phase 6 LLM orchestrator
- `NCBI_EMAIL` — required by PubMed E-utilities usage policy
- `NCBI_API_KEY` — optional, raises E-utilities rate limit from 3→10 req/sec

`.env` is gitignored — never commit real keys.

## Data Source & Reproduction

**Source:** [<-- PASTE YOUR KAGGLE DATASET URL/NAME HERE -->]

This is a mirror of the original [PTB-XL dataset](https://physionet.org/content/ptb-xl/1.0.3/) (Wagner et al., 2020), pre-split into train/valid/test and provided as six CSV files:

| File | Rows | Description |
|---|---|---|
| `train_meta.csv` | 17,441 | Demographics + diagnostic labels (NORM/MI/STTC/HYP/CD + sub-codes) + `strat_fold` |
| `train_signal.csv` | 17,441,000 | Long-format: 1 row per timestep, `ecg_id` + `channel-0`...`channel-11` (100Hz, 12-lead) |
| `valid_meta.csv` / `valid_signal.csv` | 2,193 / 2,193,000 | Same structure, validation split |
| `test_meta.csv` / `test_signal.csv` | 2,203 / 2,203,000 | Same structure, test split |

**Confirmed:** this split matches PTB-XL's canonical stratified benchmark folds — `train_fold` covers `strat_fold` 1–8, `valid` = fold 9, `test` = fold 10 — so results are directly comparable to published PTB-XL literature.

### Reproducing the processed data from raw

1. Download the dataset from the source link above.
2. Place all six CSVs into `data/raw/`.
3. Run the reshape script:
   ```powershell
   python src/data/reshape_signals.py
   ```
   This converts each split's long-format signal CSV into one `.npy` file per `ecg_id` (shape `(1000, 12)`), saved to `data/processed/signals/{split}/`, and verifies the output file count matches the meta row count for each split.

Expected output file counts: `train` = 17,441, `valid` = 2,193, `test` = 2,203.

## MLOps: DVC & MLflow

MLOps is being learned in parallel and applied selectively — only where it fits naturally into this pipeline. Full production MLOps (CI/CD, containerized deployment, live monitoring, retrain triggers) is **out of scope** here and is practiced separately.

**Applied in this project:**
- **DVC** — versions `data/raw/` and `data/processed/` so data changes are tracked alongside code. No remote is currently configured — cloning this repo gives you the code and the `.dvc` pointer files, but you'll need to re-download the raw data (see above) and re-run the reshape script rather than `dvc pull`.
- **MLflow** — tracks training runs for the per-modality and fusion models (metrics, hyperparameters, run comparisons). Runs locally via `mlflow ui` against the `mlruns/` folder.

**Explicitly not applied** (deferred to separate practice projects): model registry, Docker/API packaging, CI/CD, deployment, drift/performance monitoring, automated retrain triggers.

## Roadmap / Phase Status

8-week accelerated timeline. Status as of the last update:

- [x] **Phase 1 — Data & Baseline Setup**: PTB-XL acquired, reshaped, split-verified against canonical folds, DVC-tracked.
- [ ] **Phase 2 — Per-Modality Prediction Models**: pretrained ECG classifier + lab/EHR classifier, baseline scores.
- [ ] **Phase 3 — Multimodal Fusion**: feature-level concatenation → trained fusion classifier.
- [ ] **Phase 4 — Explainability Layer**: SHAP on the fused model.
- [ ] **Phase 5 — Evidence Retrieval**: PubMed E-utilities integration, feature → query translation.
- [ ] **Phase 6 — Report Generation**: Groq-based orchestrator, tool-calling for SHAP/retrieval lookups.
- [ ] **Phase 7 — Evaluation & Documentation**: held-out test evaluation, methodology write-up.
- [ ] **Phase 8 — Buffer, Polish & Presentation**.

## Key Risks

- **Lab/EHR modality is thin.** The current dataset's structured features are limited to demographics (age, sex, height, weight — `height` has notable missingness). This may or may not be sufficient as a genuine second "modality" — worth revisiting before Phase 2/3; the documented fallback is pairing in a separate structured dataset (e.g., UCI Heart Disease) as an explicitly stated simplifying assumption.
- **Fusion complexity** — if feature-level fusion underperforms or is unstable, fallback to late-fusion (combining independent predictions) with transparent justification.
- **Time sink risk** — Phases 5–6 (retrieval + report generation) tend to take longer than expected; reserve buffer time here if earlier phases finish ahead of schedule.
