"""Data loading: MMLU question sample and GoEmotions texts."""
from __future__ import annotations

import numpy as np
import pandas as pd
from datasets import load_dataset


def load_mmlu_sample(n: int, seed: int, max_chars: int = 1500) -> pd.DataFrame:
    """Stratified-by-subject sample of MMLU test questions.

    Returns columns: qid, subject, question, choices (list[4]), answer_idx.
    """
    ds = load_dataset("cais/mmlu", "all", split="test")
    df = ds.to_pandas()
    df["choices"] = df["choices"].apply(list)
    length = df["question"].str.len() + df["choices"].apply(lambda c: sum(len(x) for x in c))
    df = df[length <= max_chars].reset_index(drop=True)

    frac = n / len(df)
    parts = []
    for _, g in df.groupby("subject"):
        k = max(1, int(round(len(g) * frac)))
        parts.append(g.sample(n=min(k, len(g)), random_state=seed))
    out = pd.concat(parts).sample(frac=1.0, random_state=seed).head(n).reset_index(drop=True)
    out = out.rename(columns={"answer": "answer_idx"})
    out["qid"] = np.arange(len(out))
    return out[["qid", "subject", "question", "choices", "answer_idx"]]


def split_ids(qids, holdout_frac: float, seed: int) -> dict[int, str]:
    rng = np.random.default_rng(seed)
    qids = np.array(sorted(qids))
    perm = rng.permutation(len(qids))
    n_test = int(round(len(qids) * holdout_frac))
    test = set(qids[perm[:n_test]].tolist())
    return {int(q): ("test" if q in test else "train") for q in qids}


def load_goemotions(labels: list[str], n_per_label: int, n_heldout: int = 100,
                    seed: int = 0, min_single: int = 300
                    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """GoEmotions texts by label (all splits pooled). Returns (train_texts, heldout_texts).

    Prefers single-label rows. For rare labels (fewer than `min_single` single-label rows,
    e.g. nervousness) falls back to rows that contain the label and no OTHER wanted label.
    Held-out size shrinks to 25% of the pool if needed. Prints the counts used.
    'neutral' is always included.
    """
    names = None
    frames = []
    for split in ("train", "validation", "test"):
        ds = load_dataset("google-research-datasets/go_emotions", "simplified", split=split)
        names = ds.features["labels"].feature.names
        frames.append(ds.to_pandas())
    df = pd.concat(frames, ignore_index=True)
    df["names"] = df["labels"].apply(lambda xs: [names[i] for i in xs])
    df = df[df["text"].str.len().between(20, 300)]

    want = sorted(set(labels) | {"neutral"})
    train, held = {}, {}
    for lab in want:
        single = df[df["names"].apply(lambda n: n == [lab])]
        if len(single) >= min_single:
            pool, source = single, "single-label"
        else:
            others = set(want) - {lab}
            pool = df[df["names"].apply(lambda n: lab in n and not (set(n) & others))]
            source = "multi-label fallback"
        if len(pool) < 40:
            raise ValueError(f"Too few GoEmotions texts for '{lab}': {len(pool)}")
        pool = pool.sample(frac=1.0, random_state=seed)
        n_h = min(n_heldout, len(pool) // 4)
        held[lab] = pool["text"].iloc[:n_h].tolist()
        train[lab] = pool["text"].iloc[n_h:n_h + n_per_label].tolist()
        print(f"{lab:12s} {source:22s} pool={len(pool):5d} train={len(train[lab]):4d} heldout={n_h}")
    return train, held
