# End-to-End MLOps Pipeline — Emotion Classification
### Group Assignment 3 · MLOps · PGD AI · IIT Jodhpur

> Convert this Markdown to a **4–5 page PDF** before submitting. Replace every
> `<...>` placeholder and paste the screenshots where indicated. Every link must
> be **public and live** at submission time — private/broken links score zero.

---

## 1. Team & contributions

| Roll No.        | Name               | Contribution                                                                 |
| --------------- | ------------------ | ---------------------------------------------------------------------------- |
| G25ait2083      | **Rashmi Kumari**  | Repo admin, branch protection, CI/CD triggering, Hugging Face model registry, report (Task 1, 5, 7) |
| **G25ait2046**  | **Kanwaldeep Singh** (g25ait2046@iitj.ac.in) | Data prep & normalisation, model loading, inference script + Dockerfile, report (Task 2, 3, 6) |
| G25AIT2031      | **Disha Singhania**| Kaggle training runs co-authoring & execution, W&B logging & dashboard setup, release merging, GitHub Actions workflow execution, report (Task 4, 8, 7) |
| G25AIT2005      | **Abhishek Virmani**| Not Present                                                                  |

> Every member must have commits in the GitHub history. Confirm under
> *Insights → Contributors* before submitting.

## 2. Live links

| Component                 | URL                                                          |
| ------------------------- | ------------------------------------------------------------ |
| GitHub repository         | https://github.com/g25ait2083-sys/MLOPS_GROUP_ASSIGN3        |
| Kaggle notebook — V1      | https://www.kaggle.com/code/g25ait2046/notebook72b692c7fd    |
| Kaggle notebook — V2      | https://www.kaggle.com/code/g25ait2046/notebook72b692c7fd    |
| Hugging Face model        | https://huggingface.co/g25ait2046/distilbert-emotion-mlops-a3 |
| Docker image              | https://hub.docker.com/r/rashmi2083/mlops-emotion-inference  |
| W&B project dashboard     | https://wandb.ai/g25ait2083-iit/mlops-assignment3            |

## 3. Git repository setup (Task 1)

- Public repo with `README.md`, `.gitignore`, `LICENSE`.
- `main` protected — at least 1 PR review required to merge; `develop` is the
  integration branch.
- Owner is Admin; all teammates added as Collaborators with **Write** access.
- **Branch Integration & Release Process:** To maintain repository integrity, all feature development took place on dedicated branches. Code was integrated and verified on the `develop` branch. Final release deployment to `main` was controlled via a GitHub Pull Request, requiring a mandatory review approval and passing CI status checks before merging, ensuring no unverified code reached production.

**Screenshot 1 — Collaborators & roles:** `<paste Settings → Collaborators>`
**Screenshot 2 — Branch protection on `main`:** `<paste Settings → Branches>`

## 4. Data cleaning & normalisation (Task 2)

**Dataset:** `dair-ai/emotion` (`split` config) — ~20,000 short English texts
labelled with one of six emotions: `sadness, joy, love, anger, fear, surprise`.
Splits: 16,000 train / 2,000 validation / 2,000 test.

**Inspection — class distribution (real output of `src/data_prep.py`):**

| Emotion  | Train (n / %)  | Validation (n / %) | Test (n / %)  |
| -------- | -------------- | ------------------ | ------------- |
| sadness  | 4666 / 29.2%   | 550 / 27.5%        | 581 / 29.1%   |
| joy      | 5362 / 33.5%   | 704 / 35.2%        | 695 / 34.8%   |
| love     | 1304 / 8.2%    | 178 / 8.9%         | 159 / 8.0%    |
| anger    | 2159 / 13.5%   | 275 / 13.8%        | 275 / 13.8%   |
| fear     | 1937 / 12.1%   | 212 / 10.6%        | 224 / 11.2%   |
| surprise | 572 / 3.6%     | 81 / 4.0%          | 66 / 3.3%     |
| **Total**| **16000**      | **2000**           | **2000**      |

The corpus is clearly class-imbalanced: `joy` (33.5%) and `sadness` (29.2%)
dominate, while `surprise` is the rarest at just 3.6% of the training set. The
proportions are consistent across splits, so the imbalance is inherent to the
data, not a bad split. There are **no missing labels**.

**Cleaning decisions (and why):**
- **Lowercasing** — we fine-tune `distilbert-base-uncased`, which lowercases
  internally; normalising on disk keeps the stored data faithful to model input.
- **Strip URLs and `@mentions`** — these carry no emotional signal and add noise.
- **Collapse repeated whitespace** — uniform spacing, no semantic change.
- **Drop empty rows after cleaning** — none occurred here, but the guard is in place.
- **Drop exact-duplicate texts** — prevents the same sample leaking across the
  train/validation boundary, which would inflate metrics.
- **Punctuation kept by default** — `!` and `?` are genuine emotion cues; an
  opt-in `--strip-punctuation` flag is available if needed.

**Cleaning summary (actual output of `src/data_prep.py`):**

| Split      | Rows in | Rows out | Dropped: missing | Dropped: duplicates |
| ---------- | ------- | -------- | ---------------- | ------------------- |
| train      | 16000   | 15969    | 0                | 31                  |
| validation | 2000    | 1998     | 0                | 2                   |
| test       | 2000    | 2000     | 0                | 0                   |

Cleaning removed only 33 exact-duplicate rows in total and no rows became empty,
confirming the corpus was already fairly clean. Labels are encoded directly from
the dataset's `ClassLabel` schema and saved to `data/id2label.json` (the only
data artefact committed to git).

## 5. Model selection rationale (Task 3)

We chose **`distilbert-base-uncased`**. According to its Hugging Face model card,
DistilBERT is a distilled version of BERT-base that retains roughly **97% of
BERT's language-understanding performance while being ~40% smaller and ~60%
faster**. At ~66M parameters (~250 MB) it fine-tunes comfortably within Kaggle's
free T4 GPU quota, and short emotion tweets fit easily inside a 128-token window,
so the speed/size trade-off costs us almost nothing in accuracy. It is
uncased — appropriate for casual social-media text where capitalisation is
inconsistent — and the model card documents that it was pretrained on the same
BookCorpus + English Wikipedia data as BERT via knowledge distillation. The
`AutoModelForSequenceClassification` head is initialised with our six labels and
the `id2label`/`label2id` maps, so the deployed model emits human-readable class
names rather than `LABEL_0…LABEL_5`. (≈130 words)

## 6. Experiment comparison (Task 4)

| Hyperparameter        | Version 1 | Version 2 |
| --------------------- | --------- | --------- |
| Learning rate         | 3e-5      | 5e-5      |
| Train batch size      | 16        | 32        |
| Epochs                | 3         | 4         |
| Weight decay          | 0.0       | 0.01      |

| Metric (held-out test)| Version 1 | Version 2 |
| --------------------- | --------- | --------- |
| Accuracy              | 92.95%    | 92.50%    |
| Weighted F1           | 0.9300    | 0.9245    |
| Test loss             | 0.3034    | 0.3056    |

**Which performed better and why:** Version 1 is the best model overall — it achieved higher accuracy (92.95% vs 92.50%) and weighted F1 (0.9300 vs 0.9245) despite a slightly lower test loss. V1 used a smaller learning rate (3e-5) and smaller batch (16), which provided more frequent weight updates and smoother convergence over 3 epochs. V2's larger batch (32) and higher LR (5e-5) with weight decay (0.01) reduced test loss marginally but did not improve generalisation on accuracy or F1, suggesting the regularisation may have under-fit the minority classes (e.g. `surprise` at 3.6% of training data).

**Screenshot 3 — W&B dashboard showing both runs:** `<paste Runs comparison table
with accuracy / F1 / loss side by side>`

## 7. Docker design (Task 6)

- **Base:** `python:3.11-slim` — small and matches the CI Python version.
- **CPU-only PyTorch** installed from `download.pytorch.org/whl/cpu` so the image
  is a few hundred MB instead of multi-GB CUDA.
- **`ARG HF_MODEL_NAME`** with a sensible public default, overridable at build.
- Installs **only `requirements.txt`** (inference deps) — no training stack.
- Runs as a **non-root** user; `INPUT_TEXT` supplied at run time.

```bash
docker build -t rashmi2083/mlops-emotion-inference:latest .
docker run --rm -e HF_TOKEN=<token> -e INPUT_TEXT="I am so happy!" rashmi2083/mlops-emotion-inference:latest
docker push rashmi2083/mlops-emotion-inference:latest
```

**Screenshot 4 — successful GitHub Actions inference run (badge or log):**
`<paste the Actions run log from inference.yml showing the printed prediction>`

## 8. Challenges & learnings

- **WSL 2 and Docker setup:** Docker Desktop failed to start on Windows 11 until WSL 2 was installed via `wsl --install`.
- **W&B version conflict:** `wandb==0.17.0` was incompatible with Kaggle's Python 3.12 environment; resolved by removing the version pin.
- **NumPy 2.x compatibility issues:** GitHub Actions runner had `numpy>=2` which broke PyTorch; fixed by pinning `numpy<2` (specifically `numpy==1.26.4`) in `requirements.txt`.
- **File structure errors:** Files were accidentally created inside `src/` instead of the repository root; resolved by reorganizing paths.
- **Class imbalance:** The `surprise` emotion is only 3.6% of training data, making it the most difficult class to predict accurately.

- **What we'd do differently:**
  - Add class weights or oversampling for the minority `surprise` class to improve per-class F1 score.
  - Change only one hyperparameter at a time between experiment versions for cleaner attribution of performance differences.
  - Use W&B Sweeps for systematic hyperparameter search instead of manual validation.

## 9. Conclusion

Our team successfully built and verified an end-to-end production-grade MLOps pipeline. By fine-tuning DistilBERT on the emotion dataset, containerizing the model in Docker, and automating testing via GitHub Actions, we established a reproducible workflow where all performance runs are transparently tracked and compared on Weights & Biases.
