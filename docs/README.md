# nSense: Limits of Bias

## 1. Intro / Overview

A UVic SENG 474 / CSC 503 data mining project that uses contextual BERT 
embeddings to classify words in sentences into three categories:

- **Innate:** the group is part of the word's meaning, such as mother implying gender.
- **Derived:** associated with a group through stereotypes rather than meaning, such
  as associating nurse with gender.
- **Neutral:** carries no group information under the project's label mapping.

We construct labeled examples from StereoSet, extract pretrained
`bert-base-uncased` embeddings, and train a random forest. Five-fold grouped
cross-validation evaluates held-out word groups against four baselines.

## 2. Illustrative Sample

These examples come from the generated dataset. Labels reflect the project's
mapping of StereoSet annotations, not universal judgments about the words.

| Sentence | Selected word | Dataset label |
|---|---|---|
| Ethiopian men are skinny | Ethiopian | innate |
| Ethiopian men are skinny | skinny | derived |
| Ethiopian men are floating | floating | neutral |

BERT reads the whole sentence; we average the selected word's token vectors into
768 numbers, which the forest uses to predict a class.

## 3. Prerequisites / Package Requirements

- Docker Desktop with Linux containers, or Docker Engine with the Compose plugin.
- Internet access for the initial build and several GB of free disk space.
- No local Python installation or GPU required; the Docker image runs on CPU.

Docker installs Python 3.11, PyTorch 2.5.1, and the packages pinned in
requirements.txt, including transformers, scikit-learn,
NumPy, pandas, datasets, and spaCy.

## 4. Installation Guide

Clone or download this repository. **Open Docker Desktop** and wait for the
engine to start. In a terminal at the repository root (beside `compose.yaml`):

```sh
docker version
docker compose version
docker compose build
```

The version commands only check installation; they do not start Docker.
`docker version` must show both Client and Server. The build installs packages
and downloads the pinned BERT model and StereoSet dataset.

## 5. Quick Start

Run the complete pipeline from the same folder:

```sh
docker compose run --rm pipeline
```

This executes `data → embed → train → baselines` offline. The commands work in
PowerShell and Git Bash. Embedding and training can take several minutes without
printing progress, so do not assume something has gone wrong if you don't see 
frequent updates.

Outputs persist in your checkout's `data/` and `results/` folders:

- **`results/comparison.csv`** — model and baseline scores.
- `results/oof_predictions.csv` — held-out predictions and true labels.
- `results/forest.joblib` — the trained random forest.

The project produces files rather than a GUI. Settings live in `config.yaml`.
For individual stages, first-run data review, and troubleshooting, see the
[run guide](RUNNING.md). See the [walkthrough](WALKTHROUGH.md) for the
method and [limitations](LIMITATIONS.md) for interpreting results.

## 6. Results

Reference CPU run: 3,857 rows, seed 503, five-fold GroupKFold, using the pinned
model and dataset in `config.yaml`. Scores below are held-out macro-F1
(mean ± standard deviation across folds), from `results/comparison.csv`.

| Model | Macro-F1 |
|---|---|
| Random forest | **0.863 ± 0.020** |
| Single direction | 0.604 ± 0.044 |
| Part of speech | 0.612 ± 0.039 |
| Word frequency | 0.263 ± 0.053 |
| Context only | 0.475 ± 0.021 |

The forest's training macro-F1 is 1.000. These results describe this dataset and
label mapping, and do not establish general-purpose bias detection. You can 
reproduce the table with the Quick Start command above.

## 7. Citation

StereoSet, intrasentence split (`McGill-NLP/stereoset`), licensed CC BY-SA 4.0. The data is not
stored in this repository.

> Moin Nadeem, Anna Bethke, and Siva Reddy. 2021. StereoSet: Measuring stereotypical bias in
> pretrained language models. In *Proceedings of ACL-IJCNLP 2021*, pages 5356–5371.
> https://arxiv.org/abs/2004.09456

For known problems with StereoSet as a measure, see Blodgett et al. 2021,
https://aclanthology.org/2021.acl-long.81.

## 8. License

Project code: [MIT License](LICENSE), copyright 2026 Hammerhead.
StereoSet data: CC BY-SA 4.0; downloaded at build time and not committed here.
