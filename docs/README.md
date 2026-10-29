# nSense
This project is part the UVIC SENG 474 / 503 Datamining course. As a group project this repo serves as the living artifact, which stands as a monument to our journey in bias classification of embeddings using vector spaces.




# Limits of Bias: Innate / Derived / Neutral Classifier

UVic CSC503 data mining project. We sort words in sentences into three classes of bias, using
`bert-base-uncased` embeddings and a random forest.

| Class | Meaning | Example |
|---|---|---|
| innate | The group is part of the word's meaning | *girls*, *queen*, *Muslim* |
| derived | Linked to a group by stereotype, not by meaning | *soft*, *nurse* |
| neutral | Carries no group information | *fish*, *chair* |

See `docs/HANDOFF.md` for the spec and `docs/DECISIONS.md` for every judgement call.

## Run

Everything runs in a CPU-only Docker image. The pinned BERT model and the pinned StereoSet
revision are downloaded at build time, so the container runs offline.

```sh
docker build -t limits-of-bias .

# Full pipeline: data -> embed -> train -> baselines
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" limits-of-bias

# One stage (e.g. stage 1 only)
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" limits-of-bias python src/data.py
```

To run edited code without rebuilding, also mount `-v "$PWD/src:/app/src" -v "$PWD/config.yaml:/app/config.yaml"`.

Settings live in `config.yaml`: seed, model and revision, layer, axes, dataset revision,
and paths.

## Stages

| Stage | Reads | Writes |
|---|---|---|
| `src/data.py` | StereoSet (pinned revision) | `data/rows.csv`, `data/cross_axis_probe.csv`, `results/data_report.json` |
| `src/embed.py` | `data/rows.csv` | `data/embeddings.npy`, `data/embeddings_meta.json` |
| `src/train.py` | rows + embeddings | `results/` |
| `src/baselines.py` | rows + embeddings | `results/` |

`data/` and `results/` are generated and not committed. `embed.py` only needs to run once;
teammates without a GPU can work from a shared `embeddings.npy`.

## Data and citation

StereoSet, intrasentence split (`McGill-NLP/stereoset`), licensed CC BY-SA 4.0. The data is not
stored in this repository.

> Moin Nadeem, Anna Bethke, and Siva Reddy. 2021. StereoSet: Measuring stereotypical bias in
> pretrained language models. In *Proceedings of ACL-IJCNLP 2021*, pages 5356–5371.
> https://arxiv.org/abs/2004.09456

For known problems with StereoSet as a measure, see Blodgett et al. 2021,
https://aclanthology.org/2021.acl-long.81.
