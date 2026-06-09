# Emotion Classifier — End-to-End MLOps Pipeline

Fine-tune a compact transformer to classify the **emotion** expressed in a short
English text into one of six classes, then wrap the whole thing in a
production-style MLOps pipeline: reproducible data prep, GPU training on Kaggle,
experiment tracking on Weights & Biases, a containerised inference service, and
GitHub Actions for CI + on-demand inference.

> Course: MLOps · PGD AI Program · IIT Jodhpur — Group Assignment 3

---

## Task at a glance

| Component        | Choice                                                |
| ---------------- | ----------------------------------------------------- |
| Task             | Multi-class text classification (emotion detection)   |
| Dataset          | [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) (~20k samples, 6 classes) |
| Base model       | [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased) (~66M params, ~250 MB) |
| Labels           | `sadness, joy, love, anger, fear, surprise`           |
| Training         | Hugging Face `Trainer` on Kaggle (GPU T4 x2)          |
| Tracking         | Weights & Biases                                      |
| Serving          | Docker image + `src/inference.py`                     |
| Automation       | GitHub Actions (CI lint + manual inference)           |

---

## Repository layout

```
.
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt              # runtime / inference deps (used by CI + Docker + Actions)
├── requirements-dev.txt          # extra training deps (Kaggle / local dev only)
├── Dockerfile
├── .dockerignore
├── data/
│   └── id2label.json             # committed label mapping (no large data files committed)
├── src/
│   ├── __init__.py
│   ├── data_prep.py              # Task 2 — clean, normalise, build label map, save dataset
│   ├── model_utils.py            # Task 3 — load tokenizer + model with correct num_labels
│   └── inference.py              # Tasks 6/7 — load model from HF Hub, predict on INPUT_TEXT
├── notebooks/
│   └── train_emotion_kaggle.ipynb# Tasks 4/5 — two W&B-tracked runs, push best model to HF
├── report/
│   └── REPORT_TEMPLATE.md         # 4–5 page report scaffold (fill screenshots + numbers)
└── .github/
    └── workflows/
        ├── ci.yml                # Task 7.1 — flake8 lint on push to develop / PR to main
        └── inference.yml         # Task 7.2 — manual workflow_dispatch inference
```

---

## Live links (fill in before submission)

> ⚠️ Broken or private links score zero for that component.

- **GitHub Repository:** https://github.com/g25ait2083-sys/MLOPS_GROUP_ASSIGN3
- **Kaggle Notebook — Version 1:** https://www.kaggle.com/code/g25ait2046/notebook72b692c7fd
- **Kaggle Notebook — Version 2:** https://www.kaggle.com/code/g25ait2046/notebook72b692c7fd (same notebook, both runs)
- **Hugging Face Model:** https://huggingface.co/g25ait2046/distilbert-emotion-mlops-a3
- **Docker Image:** `https://hub.docker.com/r/<your-dockerhub>/mlops-a3-emotion`
- **W&B Project Dashboard:** https://wandb.ai/g25ait2046-iitjodhpur/mlops-assignment3

---

## Setup & how to run each script

### 1. Local environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate           # macOS / Linux
pip install -r requirements.txt -r requirements-dev.txt
```

### 2. Prepare the data (Task 2)

Downloads the raw dataset, inspects it, cleans/normalises the text, builds the
label mapping, and writes the prepared splits locally. Only `data/id2label.json`
is committed — the prepared parquet files are git-ignored.

```powershell
python src/data_prep.py --output-dir data/prepared
```

### 3. Train on Kaggle (Tasks 4 & 5)

Upload `notebooks/train_emotion_kaggle.ipynb` to Kaggle, enable **GPU T4 x2**,
add `WANDB_API_KEY` and `HF_TOKEN` as **Kaggle Secrets**, then Run All. It runs
two experiment versions, logs everything to W&B, and pushes the best model to the
Hugging Face Hub. Never hardcode tokens.

### 4. Run inference locally (Task 6/7)

```powershell
$env:HF_MODEL_NAME = "<your-username>/distilbert-emotion-mlops-a3"
$env:INPUT_TEXT    = "I can't stop smiling, today was wonderful!"
python src/inference.py
```

### 5. Build & run the Docker image (Task 6)

```powershell
docker build --build-arg HF_MODEL_NAME=<your-username>/distilbert-emotion-mlops-a3 -t mlops-a3-emotion:latest .
docker run --rm -e INPUT_TEXT="I am terrified of what comes next." mlops-a3-emotion:latest
```

---

## GitHub Actions

| Workflow            | Trigger                              | Purpose                |
| ------------------- | ------------------------------------ | ---------------------- |
| `ci.yml`            | push to `develop`, PR to `main`      | flake8 lint of `src/`  |
| `inference.yml`     | manual `workflow_dispatch`           | run a single prediction|

Add repository secrets `HF_TOKEN` and `WANDB_API_KEY` under
**Settings → Secrets and variables → Actions**. Training is **never** run in
Actions — only on Kaggle.

---

## Team & contributions

| Roll No.    | Name                | Primary contribution                                |
| ----------- | ------------------- | --------------------------------------------------- |
| G25ait2083  | _(admin)_           | Repo admin, branch protection                       |
| **G25ait2046** | **Kanwaldeep Singh** | Data prep + model loading + inference + Dockerfile + CI/inference workflows |
| _add_       | _add_               | _add_                                               |

> Each member must have commits visible in the GitHub history. Push your own work
> on a feature branch and open a PR into `develop`.

## License

MIT — see [LICENSE](LICENSE).

---

## 📌 Important links (for future reference)

Keep this table up to date — paste each link as soon as the resource is public.
Broken or private links score zero in the report.

### Git / GitHub
| What | Link |
| ---- | ---- |
| Repository (public) | https://github.com/g25ait2083-sys/MLOPS_GROUP_ASSIGN3 |
| `develop` branch | https://github.com/g25ait2083-sys/MLOPS_GROUP_ASSIGN3/tree/develop |
| Actions (CI + Inference) | https://github.com/g25ait2083-sys/MLOPS_GROUP_ASSIGN3/actions |

### Kaggle notebooks (set to Public)
| What | Link |
| ---- | ---- |
| Notebook — Version 1 (run-v1) | https://www.kaggle.com/code/g25ait2046/notebook72b692c7fd |
| Notebook — Version 2 (run-v2) | https://www.kaggle.com/code/g25ait2046/notebook72b692c7fd |

### Hugging Face
| What | Link |
| ---- | ---- |
| Fine-tuned model (push your own) | https://huggingface.co/g25ait2046/distilbert-emotion-mlops-a3 |
| Base model used | https://huggingface.co/distilbert-base-uncased |
| Dataset used | https://huggingface.co/datasets/dair-ai/emotion |

### Weights & Biases (set project to Public)
| What | Link |
| ---- | ---- |
| W&B project dashboard | https://wandb.ai/g25ait2046-iitjodhpur/mlops-assignment3 |
| Runs comparison (v1 vs v2) | https://wandb.ai/g25ait2046-iitjodhpur/mlops-assignment3 |

### Docker
| What | Link |
| ---- | ---- |
| Image (public registry) | `https://hub.docker.com/r/<your-dockerhub>/mlops-a3-emotion` |

### Project docs
| What | Link |
| ---- | ---- |
| Getting started / manual steps | [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) |
| Report template (→ PDF) | [report/REPORT_TEMPLATE.md](report/REPORT_TEMPLATE.md) |
| Kaggle training notebook | [notebooks/train_emotion_kaggle.ipynb](notebooks/train_emotion_kaggle.ipynb) |
