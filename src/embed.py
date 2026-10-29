"""Stage 2: data/rows.csv -> data/embeddings.npy, data/context_embeddings.npy, data/embeddings_meta.json.

e(o) for o = (s, i): BERT reads the whole sentence s; the span i only selects which token vectors
to pool.
  span vector    = mean of the tokens whose character offsets overlap [span_start, span_end)
  context vector = mean of every other token, without [CLS]/[SEP] (for the context-only baseline)
"""
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from utils import load_config, path, set_seed, write_json


def span_mask(offsets, special, start, end):
    """Tokens (not special) whose character offsets overlap [start, end)."""
    return np.array([not sp and a < end and b > start
                     for (a, b), sp in zip(offsets, special)])


def check_alignment(tokens, word, row):
    got = "".join(t.removeprefix("##") for t in tokens)
    want = word.lower().replace(" ", "")
    assert got == want, f"row {row}: span tokens {tokens} decode to {got!r}, word is {want!r}"


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    assert cfg["pooling"] == "mean", f"unsupported pooling: {cfg['pooling']}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = pd.read_csv(path(cfg, "rows"), keep_default_na=False)
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], revision=cfg["model_revision"],
                                        use_fast=True)
    model = AutoModel.from_pretrained(cfg["model_name"], revision=cfg["model_revision"])
    model.eval().to(device)

    span_vecs, ctx_vecs = [], []
    bs = cfg["batch_size"]
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            batch = rows.iloc[i:i + bs]
            enc = tok(batch["sentence"].tolist(), padding=True, return_offsets_mapping=True,
                      return_special_tokens_mask=True, return_tensors="pt")
            assert enc["input_ids"].shape[1] <= tok.model_max_length
            offsets = enc.pop("offset_mapping").tolist()
            special = enc.pop("special_tokens_mask").bool().tolist()
            out = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True)
            hidden = out.hidden_states[cfg["layer"]].float().cpu().numpy()

            for j, r in enumerate(batch.itertuples()):
                n = int(enc["attention_mask"][j].sum())
                span = span_mask(offsets[j][:n], special[j][:n], r.span_start, r.span_end)
                ctx = ~span & ~np.array(special[j][:n])
                ids = enc["input_ids"][j][:n].tolist()
                check_alignment(tok.convert_ids_to_tokens([t for t, m in zip(ids, span) if m]),
                                r.word, i + j)
                assert ctx.any(), f"row {i + j}: no context tokens outside the span"
                span_vecs.append(hidden[j, :n][span].mean(axis=0))
                ctx_vecs.append(hidden[j, :n][ctx].mean(axis=0))

    span_vecs = np.stack(span_vecs).astype(np.float32)
    ctx_vecs = np.stack(ctx_vecs).astype(np.float32)
    np.save(path(cfg, "embeddings"), span_vecs)
    np.save(path(cfg, "context_embeddings"), ctx_vecs)
    meta = {"model_name": cfg["model_name"], "model_revision": cfg["model_revision"],
            "layer": cfg["layer"], "pooling": cfg["pooling"],
            "context_pooling": "mean of non-span, non-special tokens",
            "dataset_revision": cfg["dataset_revision"], "row_count": len(rows),
            "dim": int(span_vecs.shape[1]), "device": device}
    write_json(meta, path(cfg, "embeddings_meta"))

    print(f"Alignment check: all {len(rows)} rows OK (span tokens decode to the word)")
    print(f"Span vectors {span_vecs.shape}, context vectors {ctx_vecs.shape}, device {device}")
    print(f"Wrote {path(cfg, 'embeddings')}, {path(cfg, 'context_embeddings')} "
          f"and {path(cfg, 'embeddings_meta')}")


if __name__ == "__main__":
    main()
