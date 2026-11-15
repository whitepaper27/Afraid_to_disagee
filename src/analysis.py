"""Metrics, bootstrap CIs, paired tests, calibration, monitor AUROC, figures.

Every number reported in the paper must come from tables written here.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from .directions import EMOTION_FOR
from .prompts import CONDITIONS, LETTERS, PRESSURE_CONDITIONS

N_BOOT = 2000


def boot_mean(x, n=N_BOOT, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_diff(a, b, n=N_BOOT, seed=0):
    """Difference of means of two independent samples (mean(a) - mean(b))."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    da = a[rng.integers(0, len(a), size=(n, len(a)))].mean(1)
    db = b[rng.integers(0, len(b), size=(n, len(b)))].mean(1)
    d = da - db
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ---------------------------------------------------------------- H1
def rate_table(t1: pd.DataFrame, spec: str = "none") -> pd.DataFrame:
    rows = []
    d = t1[t1["spec"] == spec]
    for cond in CONDITIONS:
        c = d[d["condition"] == cond]
        if not len(c):
            continue
        bad = c[c["kind"] == "bad"]["bad_flip"].astype(float).values
        good = c[c["kind"] == "good"]["good_update"].astype(float).values
        b, blo, bhi = boot_mean(bad)
        g, glo, ghi = boot_mean(good)
        disc, dlo, dhi = boot_diff(good, bad) if len(good) and len(bad) else (np.nan,) * 3
        rows.append({"spec": spec, "condition": cond, "n_bad": len(bad), "n_good": len(good),
                     "bad_flip": b, "bad_lo": blo, "bad_hi": bhi,
                     "good_update": g, "good_lo": glo, "good_hi": ghi,
                     "discrimination": disc, "disc_lo": dlo, "disc_hi": dhi,
                     "mean_mass1": float(c["mass1"].mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- H2
def _per_qid(t1, spec_prefix, cond, col, kind):
    d = t1[(t1["spec"].str.startswith(spec_prefix)) & (t1["condition"] == cond) & (t1["kind"] == kind)]
    return d.groupby("qid")[col].mean().astype(float)


def mediation_table(t1: pd.DataFrame) -> pd.DataFrame:
    """Per pressure condition, on turn-0-correct items:
    d_emo  = bad(none) - bad(emo_ablate)
    d_rand = bad(none) - mean_s bad(rand_ablate_s)
    effect = d_emo - d_rand = mean_q [rand_mean(q) - emo(q)]   (paired by qid)
    """
    rows = []
    for cond in [c for c in CONDITIONS if c != "neutral"] + ["neutral"]:
        for kind, col in (("bad", "bad_flip"), ("good", "good_update")):
            none = _per_qid(t1, "none", cond, col, kind)
            emo = _per_qid(t1, "emo_ablate", cond, col, kind)
            mis = _per_qid(t1, "mismatch_ablate", cond, col, kind)
            rnd = _per_qid(t1, "rand_ablate", cond, col, kind)
            q = none.index.intersection(rnd.index)
            if not len(q):
                continue
            row = {"condition": cond, "kind": kind, "n": len(q),
                   "rate_none": float(none[q].mean()), "rate_rand": float(rnd[q].mean())}
            row["d_rand"], row["d_rand_lo"], row["d_rand_hi"] = boot_mean(none[q] - rnd[q])
            for name, ser in (("emo", emo), ("mis", mis)):
                qq = q.intersection(ser.index)
                if not len(qq):
                    continue
                row[f"rate_{name}"] = float(ser[qq].mean())
                row[f"d_{name}"], row[f"d_{name}_lo"], row[f"d_{name}_hi"] = boot_mean(none[qq] - ser[qq])
                row[f"effect_{name}_vs_rand"], row[f"eff_{name}_lo"], row[f"eff_{name}_hi"] = \
                    boot_mean(rnd[qq] - ser[qq])
                if name == "emo":
                    a, b = none[qq].values > 0.5, ser[qq].values > 0.5
                    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
                    row["mcnemar_n10"], row["mcnemar_n01"] = n10, n01
                    row["mcnemar_p"] = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def addition_table(t1: pd.DataFrame) -> pd.DataFrame:
    base = t1[(t1["spec"] == "none") & (t1["condition"] == "neutral")]
    rows = []
    add = t1[t1["spec"].str.startswith("add_")]
    for spec, g in add.groupby("spec"):
        lab = spec.split("_")[1]
        for kind, col in (("bad", "bad_flip"), ("good", "good_update")):
            a = g[g["kind"] == kind].set_index("qid")[col].astype(float)
            b = base[base["kind"] == kind].set_index("qid")[col].astype(float)
            q = a.index.intersection(b.index)
            m, lo, hi = boot_mean(a[q] - b[q])
            rows.append({"spec": spec, "emotion": lab, "alpha": float(g["alpha"].iloc[0]),
                         "kind": kind, "n": len(q), "rate_add": float(a[q].mean()),
                         "rate_base": float(b[q].mean()), "delta": m, "lo": lo, "hi": hi,
                         "mean_mass1": float(g["mass1"].mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- calibration
def ece_brier(probs: np.ndarray, answer: np.ndarray, n_bins: int = 10):
    conf = probs.max(1)
    acc = (probs.argmax(1) == answer).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    onehot = np.eye(4)[answer]
    brier = float(((probs - onehot) ** 2).sum(1).mean())
    return float(ece), brier, float(acc.mean())


def calibration_table(t0: pd.DataFrame, t1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    p0 = t0[[f"p0_{L}" for L in LETTERS]].values
    e, b, a = ece_brier(p0, t0["answer_idx"].values)
    rows.append({"stage": "turn0", "condition": "-", "ece": e, "brier": b, "accuracy": a, "n": len(t0)})
    d = t1[t1["spec"] == "none"]
    for cond in CONDITIONS:
        c = d[d["condition"] == cond]
        if not len(c):
            continue
        e, b, a = ece_brier(c[[f"p1_{L}" for L in LETTERS]].values, c["answer_idx"].values)
        rows.append({"stage": "turn1", "condition": cond, "ece": e, "brier": b, "accuracy": a, "n": len(c)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- H4 monitor
def monitor_table(t0: pd.DataFrame, t1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d = t1[(t1["spec"] == "none") & (t1["kind"] == "bad")].merge(
        t0[["qid"] + [f"p0_{L}" for L in LETTERS]], on="qid")
    d["p0_correct"] = d.apply(lambda r: r[f"p0_{LETTERS[r['answer_idx']]}"], axis=1)
    for cond in [c for c in CONDITIONS if EMOTION_FOR.get(c)]:
        lab = EMOTION_FOR[cond]
        c = d[d["condition"] == cond]
        y = c["bad_flip"].astype(int).values
        col = f"proj_{lab}"
        if col not in c or y.min() == y.max():
            continue
        x_proj = c[col].values
        x_conf = 1 - c["p0_correct"].values
        auc_proj = roc_auc_score(y, x_proj)
        auc_conf = roc_auc_score(y, x_conf)
        X = np.c_[x_proj, x_conf]
        k = int(min(5, np.bincount(y).min()))
        auc_both = (roc_auc_score(y, cross_val_predict(LogisticRegression(), X, y, cv=k,
                                                        method="predict_proba")[:, 1])
                    if k >= 2 else np.nan)
        rows.append({"condition": cond, "emotion": lab, "n": len(y), "n_flip": int(y.sum()),
                     "auroc_projection": auc_proj, "auroc_turn0_conf": auc_conf,
                     "auroc_combined_cv": auc_both})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- H3 erase vs override
def lens_summary(t1: pd.DataFrame, lens: pd.DataFrame, n_layers_hidden: int) -> pd.DataFrame:
    """For bad flips (spec none): P(correct) by depth band at turn 1 vs turn 0, same items."""
    flips = t1[(t1["spec"] == "none") & (t1["bad_flip"])][["qid", "condition"]]
    rows = []
    l0 = lens[lens["condition"] == "turn0"]
    for cond, g in flips.groupby("condition"):
        qs = set(g["qid"])
        l1 = lens[(lens["condition"] == cond) & (lens["turn"] == 1) & lens["qid"].isin(qs)]
        base = l0[l0["qid"].isin(qs)]
        for name, lo, hi in (("early", 0.0, 0.33), ("middle", 0.33, 0.66),
                             ("late", 0.66, 0.95), ("final", 0.95, 1.01)):
            sel = lambda df: df[(df["layer"] >= lo * n_layers_hidden) & (df["layer"] < hi * n_layers_hidden)]
            rows.append({"condition": cond, "band": name, "n_items": len(qs),
                         "p_correct_turn1": float(sel(l1)["p_correct"].mean()),
                         "p_correct_turn0": float(sel(base)["p_correct"].mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
def fig1_mediation(t1, path):
    present = set(t1["condition"])
    conds = [c for c in CONDITIONS if c != "neutral" and c in present]
    specs = [("none", "No intervention"), ("emo_ablate", "Matched emotion removed"),
             ("rand_ablate", "Random direction removed"), ("mismatch_ablate", "Mismatched emotion removed")]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    w = 0.2
    for k, (sp, label) in enumerate(specs):
        ys, los, his = [], [], []
        for cond in conds:
            s = _per_qid(t1, sp, cond, "bad_flip", "bad")
            m, lo, hi = boot_mean(s.values) if len(s) else (np.nan,) * 3
            ys.append(m); los.append(m - lo if len(s) else 0); his.append(hi - m if len(s) else 0)
        x = np.arange(len(conds)) + (k - 1.5) * w
        ax.bar(x, ys, w, yerr=[los, his], capsize=3, label=label)
    ax.set_xticks(np.arange(len(conds)))
    ax.set_xticklabels([c.replace("_", "\n") for c in conds])
    ax.set_ylabel("Abandoned a correct answer (rate)")
    ax.set_title("Does removing the matched emotion direction reduce caving under pressure?")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig2_discrimination(rates, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    r = rates.set_index("condition").loc[[c for c in CONDITIONS if c in set(rates["condition"])]]
    x = np.arange(len(r))
    ax.bar(x - 0.2, r["bad_flip"], 0.4, yerr=[r["bad_flip"] - r["bad_lo"], r["bad_hi"] - r["bad_flip"]],
           capsize=3, label="Bad flips (abandons correct)")
    ax.bar(x + 0.2, r["good_update"], 0.4,
           yerr=[r["good_update"] - r["good_lo"], r["good_hi"] - r["good_update"]],
           capsize=3, label="Good updates (fixes wrong)")
    ax.set_xticks(x); ax.set_xticklabels([c.replace("_", "\n") for c in r.index])
    ax.set_ylabel("Rate"); ax.legend(frameon=False)
    ax.set_title("Pushback: caving vs. correcting, by pressure type")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig3_lens(t1, lens, path):
    flips = t1[(t1["spec"] == "none") & (t1["bad_flip"])][["qid", "condition"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    l0 = lens[lens["condition"] == "turn0"]
    allq = set(flips["qid"])
    if allq:
        b = l0[l0["qid"].isin(allq)].groupby("layer")["p_correct"].mean()
        ax.plot(b.index, b.values, "k--", label="Turn 0 (before pushback)")
    for cond, g in flips.groupby("condition"):
        l1 = lens[(lens["condition"] == cond) & (lens["turn"] == 1) & lens["qid"].isin(set(g["qid"]))]
        m = l1.groupby("layer")["p_correct"].mean()
        ax.plot(m.index, m.values, label=f"{cond} (n={g['qid'].nunique()})")
    ax.set_xlabel("Layer"); ax.set_ylabel("Logit-lens P(correct answer)")
    ax.set_title("Flipped items: is the correct answer still readable mid-network?")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig4_directions(val, band, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    for lab, g in val.groupby("label"):
        ax.plot(g["layer"], g["auroc"], label=lab)
    if band:
        ax.axvspan(band[0] + 1, band[1] + 1, alpha=0.15, color="gray", label="intervention band")
    ax.set_xlabel("Hidden-state index (layer + 1)"); ax.set_ylabel("Held-out AUROC vs neutral")
    ax.set_title("Emotion directions separate held-out emotional text from neutral text")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def write_all(cfg, t0, t1, lens, val, band, n_layers_hidden):
    tdir = os.path.join(cfg.out_dir, "tables"); fdir = os.path.join(cfg.out_dir, "figures")
    os.makedirs(tdir, exist_ok=True); os.makedirs(fdir, exist_ok=True)
    rates = rate_table(t1, "none")
    tables = {"rates_none": rates,
              "mediation": mediation_table(t1),
              "addition": addition_table(t1),
              "calibration": calibration_table(t0, t1),
              "monitor": monitor_table(t0, t1),
              "lens_summary": lens_summary(t1, lens, n_layers_hidden)}
    for name, df in tables.items():
        df.to_csv(os.path.join(tdir, f"{name}.csv"), index=False)
    fig1_mediation(t1, os.path.join(fdir, "fig1_mediation.png"))
    fig2_discrimination(rates, os.path.join(fdir, "fig2_discrimination.png"))
    fig3_lens(t1, lens, os.path.join(fdir, "fig3_lens.png"))
    if val is not None:
        fig4_directions(val, band, os.path.join(fdir, "fig4_directions.png"))
    return tables
