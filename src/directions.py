"""Emotion directions: mean-difference (emotion - neutral) per layer, validated on held-out texts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from .prompts import build_emotion_probe
from .readout import forward_last

# Pressure condition -> matched GoEmotions label (None = no matched emotion).
EMOTION_FOR = {
    "neutral": None,
    "threat": "fear",
    "urgency": "nervousness",
    "reward": "excitement",
    "praise": "admiration",
    "control_nosuggest": "fear",
    "authority": None,          # exploratory: behavioral + random control only
    "boss_deadline": "fear",    # exploratory
}
# Mismatched-emotion control: opposite valence.
MISMATCH_FOR = {
    "threat": "joy",
    "urgency": "joy",
    "reward": "fear",
    "praise": "fear",
    "control_nosuggest": "joy",
    "boss_deadline": "joy",
}
EMOTION_LABELS = sorted({v for v in EMOTION_FOR.values() if v} | set(MISMATCH_FOR.values()))


@torch.no_grad()
def last_token_states(model, tok, texts: list[str], batch_size: int = 16) -> torch.Tensor:
    outs = []
    prompts = [build_emotion_probe(tok, t) for t in texts]
    for i in tqdm(range(0, len(prompts), batch_size), leave=False):
        _, h = forward_last(model, tok, prompts[i:i + batch_size], want_hidden=True)
        outs.append(h.cpu())
    return torch.cat(outs)  # [N, L+1, d]


def extract(model, tok, train_texts: dict[str, list[str]], batch_size: int = 16):
    """Returns (dirs {label: Tensor[L+1,d] unit}, means {label: Tensor[L+1,d]})."""
    means = {lab: last_token_states(model, tok, texts, batch_size).mean(0)
             for lab, texts in train_texts.items()}
    neutral = means["neutral"]
    dirs = {}
    for lab, m in means.items():
        if lab == "neutral":
            continue
        d = m - neutral
        dirs[lab] = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return dirs, means


def validate(model, tok, dirs, held_texts, batch_size: int = 16) -> pd.DataFrame:
    """AUROC per (label, layer): projection on d separates label texts from neutral texts."""
    h_neu = last_token_states(model, tok, held_texts["neutral"], batch_size)
    rows = []
    for lab, d in dirs.items():
        h_lab = last_token_states(model, tok, held_texts[lab], batch_size)
        for layer in range(d.shape[0]):
            s_lab = (h_lab[:, layer] @ d[layer]).numpy()
            s_neu = (h_neu[:, layer] @ d[layer]).numpy()
            y = np.r_[np.ones(len(s_lab)), np.zeros(len(s_neu))]
            rows.append({"label": lab, "layer": layer,
                         "auroc": roc_auc_score(y, np.r_[s_lab, s_neu])})
    return pd.DataFrame(rows)


def suggest_band(val: pd.DataFrame, n_decoder_layers: int, labels: list[str],
                 lo_frac: float = 0.25, hi_frac: float = 0.8) -> tuple[int, int]:
    """Contiguous window (~1/4 depth) of decoder layers with the best mean AUROC.

    Decoder layer i corresponds to hidden-state index i+1.
    """
    width = max(2, n_decoder_layers // 4)
    m = (val[val["label"].isin(labels)].groupby("layer")["auroc"].mean())
    lo, hi = int(n_decoder_layers * lo_frac), int(n_decoder_layers * hi_frac)
    best, best_start = -1.0, lo
    for start in range(lo, hi - width + 1):
        score = np.mean([m.get(i + 1, 0.0) for i in range(start, start + width)])
        if score > best:
            best, best_start = score, start
    return best_start, best_start + width - 1


def random_dirs(n_hidden_rows: int, d_model: int, n: int, seed: int) -> list[torch.Tensor]:
    """n random unit-direction sets, each Tensor[L+1, d] (independent per layer)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        r = torch.randn(n_hidden_rows, d_model, generator=g)
        out.append(r / r.norm(dim=-1, keepdim=True))
    return out
