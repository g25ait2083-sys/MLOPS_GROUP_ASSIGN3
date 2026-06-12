================================================================================
PRODUCTION-GRADE MLOPS PIPELINE: DISTRIBUTED EMOTION CLASSIFICATION
================================================================================
Course: MLOps (PGD AI Program) | IIT Jodhpur
Project: Group Assignment 3
Classification Domain: Multi-Class Short Text Emotion Detection

--------------------------------------------------------------------------------
1. ARCHITECTURAL OVERVIEW & CORE STACK
--------------------------------------------------------------------------------
This project establishes a robust, end-to-end MLOps pipeline designed to fine-tune 
a compact transformer architecture for real-time text emotion classification, 
followed by enterprise-grade artifact tracking, containerization, and automated 
CI/CD testing workflows.

Core Components:
* Target Objective: Map short English phrases to 6 distinct emotional expressions.
* Target Labels: sadness, joy, love, anger, fear, surprise
* Primary Dataset: dair-ai/emotion (~20,000 samples)
* Deep Learning Model: distilbert-base-uncased (~66M parameters, ~250 MB disk footprint)
* Training Compute: Kaggle Distributed Environment (Dual T4 GPU Accelerator Setup)
* Experiment Tracking: Weights & Biases (W&B) Workspace
* Deployment Form: Containerized Lightweight Docker Service
* Orchestration & CI: GitHub Actions Automation (Syntax Linting & Remote Dispatch)

--------------------------------------------------------------------------------
2. SYSTEM INFRASTRUCTURE / REPOSITORY TREE
--------------------------------------------------------------------------------
.
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt              # Standard production/inference runtime dependencies
├── requirements-dev.txt          # Supplemental libraries for local development & training
├── Dockerfile
├── .dockerignore
├── data/
│   └── id2label.json             # Serialized category mapping (large data binaries excluded)
├── src/
│   ├── __init__.py
│   ├── data_prep.py              # Phase 2: Cleansing, normalization, & dataset parsing
│   ├── model_utils.py            # Phase 3: Tokenizer instantiation & configuration loader
│   └── inference.py              # Phase 6/7: Remote artifact retrieval & active inference
├── notebooks/
│   └── train_emotion_kaggle.ipynb# Phase 4/5: Double-run tracked training & HF registry push
├── report/
│   └── REPORT_TEMPLATE.md         # Narrative breakdown and empirical results template
└── .github/
    └── workflows/
        ├── ci.yml                # Phase 7.1: Automated static code checks via flake8
        └── inference.yml         # Phase 7.2: On-demand pipeline verification run

--------------------------------------------------------------------------------
3. DEPLOYMENT & EXECUTION RUNBOOK
--------------------------------------------------------------------------------

[STAGE A] Local Virtual Environment Setup
----------------------------------------
Isolate dependencies and instantiate the execution context:
  $ python -m venv .venv
  
  # For Windows Environment (PowerShell):
  $ .\.venv\Scripts\Activate.ps1
  
  # For UNIX / macOS Environments:
  $ source .venv/bin/activate
  
  # Sync dependencies:
  $ pip install -r requirements.txt -r requirements-dev.txt

[STAGE B] Pre-processing & Feature Engineering (Phase 2)
-------------------------------------------------------
Ingest raw text, execute normalization protocols, compile categorical indexes, 
and export local partitions (Parquet partitions are git-ignored; only the 
generated mapping dictionary is persisted).
  $ python src/data_prep.py --output-dir data/prepared

[STAGE C] Accelerated Training on Cloud Compute (Phase 4 & 5)
------------------------------------------------------------
1. Import the file `notebooks/train_emotion_kaggle.ipynb` into Kaggle.
2. Adjust configuration to enable dual NVIDIA T4 GPU support.
3. Inject structural environment tokens (`WANDB_API_KEY` and `HF_TOKEN`) as encrypted Kaggle Secrets.
4. Execute full notebook compilation. The script automatically orchestrates dual 
   experimental runs, pipes live parameters to W&B, and commits the top-performing 
   weights directly to Hugging Face Hub. (Do not hardcode credentials).

[STAGE D] Native Command-Line Inference (Phase 6/7)
--------------------------------------------------
Define target model parameters and input data strings inside your shell context, then invoke the prediction worker:
  # Environment Configuration (Windows PowerShell):
  $ $env:HF_MODEL_NAME = "g25ait2046/distilbert-emotion-mlops-a3"
  $ $env:INPUT_TEXT    = "I can't stop smiling, today was wonderful!"
  $ python src/inference.py

[STAGE E] Containerized Microservice Build & Execution (Phase 6)
--------------------------------------------------------------
Compile the isolated runtime image and evaluate model performance under isolation:
  $ docker build --build-arg HF_MODEL_NAME=g25ait2046/distilbert-emotion-mlops-a3 -t mlops-a3-emotion:latest .
  $ docker run --rm -e INPUT_TEXT="I am terrified of what comes next." mlops-a3-emotion:latest

--------------------------------------------------------------------------------
4. AUTOMATION ENGINE (GITHUB ACTIONS)
--------------------------------------------------------------------------------
Workflow Configurations:
* ci.yml
  - Triggers: Push events to 'develop' branch, Pull Requests targeting 'main'.
  - Function: Static code analysis & compliance using flake8 on 'src/' modules.
* inference.yml
  - Triggers: Manual triggers via GitHub 'workflow_dispatch'.
  - Function: Executes live ad-hoc inference tests on a running instance.

Important: Ensure repository secrets (`HF_TOKEN` and `WANDB_API_KEY`) are fully 
provisioned under Settings -> Secrets and variables -> Actions prior to manual runs. 
Training pipelines are restricted exclusively to Kaggle infrastructure.

--------------------------------------------------------------------------------
5. ENGINEERING TEAM & PROJECT CONTRIBUTIONS
--------------------------------------------------------------------------------
Roll Number     | Contributor Name    | Defined Operational Focus Areas
----------------|---------------------|-----------------------------------------
G25ait2083      | Rashmi Kumari       | Repo setup, branch protection, CI/CD triggering (Task 1, 7, 5)
G25ait2046      | Kanwaldeep Singh    | Data prep, model loading, Docker design (Task 2, 3, 6)
G25ait2031      | Disha Singhania     | Kaggle training runs, W&B experiment tracking (Task 4, 8)

*Note: All development must be traceable to individual contributions via the commit graph. 
Developers are required to implement features on a dedicated branch, pushing to 
remote and generating Pull Requests into 'develop'.

--------------------------------------------------------------------------------
6. MASTER DIRECTORY & VERIFIED PROJECT LINKS
--------------------------------------------------------------------------------
Ensure all endpoints listed below remain public and accessible. Unreachable resource 
links will automatically result in a score penalty.

[Source Code Control]
* Main Git Repository: 
  https://github.com/g25ait2083-sys/MLOPS_GROUP_ASSIGN3
* Active Development Branch (develop): 
  https://github.com/g25ait2083-sys/MLOPS_GROUP_ASSIGN3/tree/develop
* Automation Suites & GitHub Actions: 
  https://github.com/g25ait2083-sys/MLOPS_GROUP_ASSIGN3/actions

[Compute Notebooks (Kaggle)]
* Execution Run - Iteration 01: 
  https://www.kaggle.com/code/g25ait2046/notebook72b692c7fd
* Execution Run - Iteration 02: 
  https://www.kaggle.com/code/g25ait2046/notebook72b692c7fd

[Model & Data Registries]
* Production Fine-Tuned Weights (Hugging Face): 
  https://huggingface.co/g25ait2046/distilbert-emotion-mlops-a3
* Vanilla Base Transformer Architecture: 
  https://huggingface.co/distilbert-base-uncased
* Original Training Corpus (dair-ai): 
  https://huggingface.co/datasets/dair-ai/emotion

[Experiment Metrics Workspace]
* Analytical Dashboard (W&B): 
  https://wandb.ai/g25ait2046-iitjodhpur/mlops-assignment3
* Iterative Run Comparisons (v1 vs v2): 
  https://wandb.ai/g25ait2046-iitjodhpur/mlops-assignment3

[Container Registry]
* Distributed Docker Hub Repository: 
  https://hub.docker.com/r/rashmi2083/mlops-emotion-inference

[Internal Project Documentation]
* Comprehensive Onboarding Runbook: 
  docs/GETTING_STARTED.md
* Academic Evaluation Report Scaffold: 
  report/REPORT_TEMPLATE.md
* Native Kaggle Pipeline Script: 
  notebooks/train_emotion_kaggle.ipynb

--------------------------------------------------------------------------------
7. LICENSING
--------------------------------------------------------------------------------
Distributed openly under the provisions of the MIT License. Reference the LICENSE 
binary file for complete legal guidelines.
================================================================================
