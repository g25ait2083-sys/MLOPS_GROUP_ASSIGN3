# Getting started — what to run where

This scaffold contains all the **code** for the assignment. The steps below are
the **account/manual actions** you (and your team) still need to do — they need
your GitHub, Kaggle, Hugging Face, Docker Hub and W&B logins, so they can't be
scripted here.

---

## A. Push this scaffold to the repo (Task 1)

From inside this `MLOPS_GROUP_ASSIGN3/` folder:

```powershell
git init
git branch -M main
git remote add origin https://github.com/g25ait2083-sys/MLOPS_GROUP_ASSIGN3.git
git add .
git commit -m "Scaffold MLOps emotion-classification pipeline"
# The repo already has a README commit on main, so pull/rebase first:
git pull origin main --allow-unrelated-histories --rebase
git push origin main
```

> Tip: so your contributions show up under **G25ait2046**, set your git identity
> to the email tied to that GitHub account *before* committing:
> ```powershell
> git config user.name  "<your name>"
> git config user.email "<email on your GitHub account>"
> ```

Then create and push the `develop` branch and do real work there via PRs:

```powershell
git checkout -b develop
git push -u origin develop
```

### Branch protection & collaborators (admin: g25ait2083)
- **Settings → Branches → Add rule** for `main`: enable *Require a pull request
  before merging* → *Require approvals: 1*. → Screenshot for the report.
- **Settings → Collaborators**: add each teammate (incl. G25ait2046) with
  **Write** access. → Screenshot for the report.

### Repository secrets (Task 7.3)
- **Settings → Secrets and variables → Actions → New repository secret**: add
  `HF_TOKEN` and `WANDB_API_KEY`. Never commit tokens.

---

## B. Train on Kaggle (Tasks 4 & 5)

1. Kaggle → Create → Notebook → **File → Import Notebook** →
   `notebooks/train_emotion_kaggle.ipynb`.
2. **Settings → Accelerator → GPU T4 x2**.
3. **Add-ons → Secrets**: add `WANDB_API_KEY` and `HF_TOKEN`.
4. Edit the `HF_MODEL_REPO` value in the config cell to
   `your-username/distilbert-emotion-mlops-a3`.
5. **Run All.** It runs `run-v1` and `run-v2`, logs to W&B, and pushes the best
   model to Hugging Face.
6. **Save Version** and set the notebook visibility to **Public**; copy the link.

> To submit two *separate* Kaggle notebook links, you can duplicate the notebook
> and leave only one run active in each, or share the same notebook twice — but
> both runs must appear in W&B regardless.

---

## C. Make models/dashboards public (Tasks 5 & 8)

- **Hugging Face**: open the model repo → Settings → set visibility **Public**.
- **W&B**: project → Settings → set project visibility **Public**. Use the
  **Runs comparison table** to show Accuracy / F1 / Loss for both runs; screenshot.

---

## D. Docker image (Task 6)

```powershell
docker build --build-arg HF_MODEL_NAME=your-username/distilbert-emotion-mlops-a3 -t mlops-a3-emotion:latest .
docker run --rm -e INPUT_TEXT="I am thrilled about the results!" mlops-a3-emotion:latest

# Push to a public registry:
docker login
docker tag mlops-a3-emotion:latest your-dockerhub/mlops-a3-emotion:latest
docker push your-dockerhub/mlops-a3-emotion:latest
```

---

## E. Trigger GitHub Actions (Task 7)

- **CI** runs automatically when you push to `develop` or open a PR into `main`.
- **Inference**: Actions tab → *Inference* → **Run workflow** → enter text (and
  optionally your HF model id). Screenshot the successful run / copy the log for
  the report.

---

## F. Report (5 marks)

Fill in `report/REPORT_TEMPLATE.md`, paste all screenshots and the four
experiment/log items, export to a 4–5 page PDF, and submit via the course portal.

---

## Originality / anti-plagiarism notes
- All scripts here were written for this project; cite the dataset
  (`dair-ai/emotion`) and base model (`distilbert-base-uncased`) in the report.
- Do **not** copy another group's notebook, README, or report text. Your W&B
  runs, Kaggle notebook, and HF model will be under your own accounts.
- The `data_prep.py` cleaning choices and the model-selection paragraph are
  written in your own words — keep them that way if you edit.
