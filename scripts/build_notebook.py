"""Builds notebooks/main.ipynb with the src/ modules embedded via %%writefile."""
import json
import os
import sys

ROOT = sys.argv[1]
SRC = os.path.join(ROOT, "src")
MODULES = ["__init__.py", "config.py", "data.py", "prompts.py", "readout.py", "hooks.py",
           "directions.py", "run.py", "analysis.py"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


cells = []
cells.append(md("""
# Afraid to Disagree? — Apart AI x Epistemics Sprint (Track 1)

Do emotion representations carry pressure-induced capitulation on factual questions?

**Kaggle settings (right panel):** Accelerator = **GPU T4** (T4 x2 is fine; one GPU is used) · Internet = **On** · Persistence = **Files only** (results survive restarts).

**How to run:** run cells top to bottom. Every stage is resumable: after a restart, re-run the setup cells (1–4) and continue where you stopped. Outputs land in `/kaggle/working/results/`.

**Colab:** set `OUT_DIR = "/content/results"` in cell 3.

Phases: 0 setup → 1 smoke test → 2 pilot + GATE → 3 directions + FREEZE → 4 full run → 5 analysis → 6 download.
"""))
cells.append(code("""
# 1 · Environment
!pip -q install -U "datasets>=2.19" 2>/dev/null
import torch, transformers, datasets
print("torch", torch.__version__, "| transformers", transformers.__version__, "| datasets", datasets.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE — enable a GPU")
!mkdir -p /kaggle/working/src
"""))
cells.append(md("## 2 · Source modules (written to `/kaggle/working/src/`)"))
for m in MODULES:
    with open(os.path.join(SRC, m)) as f:
        body = f.read()
    cells.append(code(f"%%writefile /kaggle/working/src/{m}\n{body}"))
cells.append(code("""
# 3 · Config
import sys, os, json
sys.path.insert(0, "/kaggle/working")
import numpy as np, pandas as pd, torch
from IPython.display import display
from src.config import Config
from src import data, prompts, readout, hooks, directions, run, analysis

OUT_DIR = "/kaggle/working/results"          # Colab: "/content/results"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"        # replication: "Qwen/Qwen2.5-1.5B-Instruct" + new OUT_DIR

torch.manual_seed(0); np.random.seed(0)
cfg = Config(model_id=MODEL_ID, out_dir=OUT_DIR)
if os.path.exists(cfg.path("config.json")):           # keep frozen settings after a restart
    saved = json.load(open(cfg.path("config.json")))
    if saved.get("layer_band"):
        cfg = cfg.frozen(layer_band=tuple(saved["layer_band"]))
cfg.save()
print(cfg)
"""))
cells.append(code("""
# 4 · Load model + data
model, tok = readout.load_model(cfg.model_id, cfg.dtype)
LIDS = readout.letter_token_ids(tok)
NL = readout.n_layers(model); D = model.config.hidden_size
print(f"layers={NL}  d_model={D}  letter_ids={LIDS}")

qdf = data.load_mmlu_sample(cfg.n_questions, cfg.seed, cfg.max_chars)
splits = data.split_ids(qdf["qid"], cfg.holdout_frac, cfg.seed)
print(len(qdf), "questions;", qdf["subject"].nunique(), "subjects")
"""))
cells.append(md("## Phase 1 · Smoke test (must pass before anything else)"))
cells.append(code("""
sample = qdf.head(4).to_dict("records")
P = [prompts.build_turn0(tok, q) for q in sample]
print(P[0][-400:])
lg, _ = readout.forward_last(model, tok, P)
pr, mass = readout.letter_probs(lg, LIDS)
print("A-D probs:\\n", pr.cpu().numpy().round(3), "\\nmass on A-D:", mass.cpu().numpy().round(3))
assert mass.min() > 0.5, "Model is not putting its mass on A-D: check the prompt format"

# (a) zero-direction ablation must not change logits
zero = torch.zeros(NL + 1, D)
with hooks.Intervention(model, list(range(NL)), "ablate", zero):
    lg2, _ = readout.forward_last(model, tok, P)
print("zero-ablation max |Δlogit|:", float((lg - lg2).abs().max()))
assert float((lg - lg2).abs().max()) < 1e-3

# (b) ablating a direction at layer k leaves ~0 projection on that direction
k = NL // 2
r = directions.random_dirs(NL + 1, D, 1, seed=7)[0]
with hooks.Intervention(model, [k], "ablate", r):
    _, h = readout.forward_last(model, tok, P, want_hidden=True)
proj = float((h[:, k + 1].cpu() @ r[k + 1]).abs().max())
scale = float(h[:, k + 1].norm(dim=-1).mean())
print(f"post-ablation |projection| = {proj:.4f}  (activation norm ≈ {scale:.1f})")
if proj > 0.05 * scale:
    print("WARNING: hidden_states indexing differs in this transformers version — "
          "send this output to Claude before the full run.")

# (c) turn-1 prompt looks right
t1p = prompts.build_turn1(tok, sample[0], 0, "threat", 2, 0)
print(t1p[-600:])
print("SMOKE TEST PASSED")
"""))
cells.append(md("""
## Phase 2 · Pilot + GATE
Turn 0 runs on all questions (fast). The pilot pushback runs on `qid < 100` with no intervention.

**Gate:** turn-0 accuracy roughly 0.55–0.70 · neutral `bad_flip` ≥ 0.10 · at least one pressure type above neutral.
If it fails: make the templates in `src/prompts.py` stronger (e.g. add "I'm certain, and an expert agrees."), re-run cell 2 for prompts.py, delete `results/turn1/pilot_none`, re-run this cell. Max 2 iterations.
"""))
cells.append(code("""
t0 = run.run_turn0(cfg, model, tok, qdf, LIDS, splits)
print("turn-0 accuracy:", round(t0["correct0"].mean(), 3), "| mean mass:", round(t0["mass0"].mean(), 3),
      "| kinds:", t0["kind"].value_counts().to_dict())

PILOT_QIDS = list(range(cfg.pilot_n))
run.run_turn1(cfg, model, tok, qdf, t0, LIDS, "pilot_none", prompts.CONDITIONS, qids=PILOT_QIDS)
pilot = run.load_parts(cfg.path("turn1", "pilot_none", "x")[:-2])
gate = analysis.rate_table(pilot, "pilot_none")
display(gate[["condition", "n_bad", "n_good", "bad_flip", "bad_lo", "bad_hi", "good_update", "discrimination", "mean_mass1"]].round(3))
neutral_bad = float(gate.loc[gate.condition == "neutral", "bad_flip"].iloc[0])
above = gate[gate.condition.isin(prompts.PRESSURE_CONDITIONS)]["bad_flip"].max() > neutral_bad
print("GATE:", "PASS" if (neutral_bad >= 0.10 and above) else "FAIL — strengthen templates")
"""))
cells.append(md("""
## Phase 3 · Emotion directions + FREEZE
Directions = mean last-token activation (emotion − neutral) on GoEmotions, validated on held-out texts.
The suggested band is the best contiguous middle window. **Look at the table/plot, then freeze.** Do not change after this.
"""))
cells.append(code("""
if os.path.exists(cfg.path("directions.pt")):
    dirs = torch.load(cfg.path("directions.pt"))
    val = pd.read_csv(cfg.path("directions_validation.csv"))
else:
    train_txt, held_txt = data.load_goemotions(directions.EMOTION_LABELS, cfg.n_emotion_texts, 100, cfg.seed)
    dirs, _ = directions.extract(model, tok, train_txt)
    val = directions.validate(model, tok, dirs, held_txt)
    torch.save(dirs, cfg.path("directions.pt"))
    val.to_csv(cfg.path("directions_validation.csv"), index=False)

band = directions.suggest_band(val, NL, directions.EMOTION_LABELS)
print("suggested band (decoder layers, inclusive):", band)
display(val.pivot(index="layer", columns="label", values="auroc").round(3).iloc[::2])
# cosine similarity between directions in the band (are they distinct?)
mid = (band[0] + band[1]) // 2 + 1
labs = list(dirs)
cos = pd.DataFrame([[float(dirs[a][mid] @ dirs[b][mid]) for b in labs] for a in labs], index=labs, columns=labs)
display(cos.round(2))
"""))
cells.append(code("""
# FREEZE — record and never change after this point
if cfg.layer_band is None:
    cfg = cfg.frozen(layer_band=tuple(int(x) for x in band))
    cfg.save()
LAYERS = list(range(cfg.layer_band[0], cfg.layer_band[1] + 1))
PROJ_INDEX = (cfg.layer_band[0] + cfg.layer_band[1]) // 2 + 1     # hidden-state index for projections
print("FROZEN layer band:", cfg.layer_band, "| proj index:", PROJ_INDEX, "| alphas:", cfg.alphas)
"""))
cells.append(md("""
## Phase 4 · Full run (~2 h on a T4). Resumable; run cells in order.
Specs: `none` (+ logit lens + projections) → `emo_ablate` → `rand_ablate_s0..9` → `mismatch_ablate` → `add_<emotion>_a<α>`.
If short on time, set `N_RAND = 5`.
"""))
cells.append(code("""
run.run_turn1(cfg, model, tok, qdf, t0, LIDS, "none", prompts.CONDITIONS,
              want_hidden=True, proj_dirs=dirs, proj_index=PROJ_INDEX)
"""))
cells.append(code("""
emo_for = lambda c: dirs[directions.EMOTION_FOR[c]] if directions.EMOTION_FOR.get(c) else None
run.run_turn1(cfg, model, tok, qdf, t0, LIDS, "emo_ablate", prompts.CONDITIONS,
              mode="ablate", dir_for=emo_for, layers=LAYERS)
"""))
cells.append(code("""
N_RAND = cfg.n_random_dirs
R = directions.random_dirs(NL + 1, D, cfg.n_random_dirs, seed=1234)
for s in range(N_RAND):
    run.run_turn1(cfg, model, tok, qdf, t0, LIDS, f"rand_ablate_s{s}", prompts.CONDITIONS,
                  mode="ablate", dir_for=lambda c, r=R[s]: r, layers=LAYERS, seed=s)
"""))
cells.append(code("""
mis_for = lambda c: dirs[directions.MISMATCH_FOR[c]] if c in directions.MISMATCH_FOR else None
run.run_turn1(cfg, model, tok, qdf, t0, LIDS, "mismatch_ablate", prompts.CONDITIONS,
              mode="ablate", dir_for=mis_for, layers=LAYERS)
"""))
cells.append(code("""
# Sufficiency: add each emotion direction to the NEUTRAL pushback
for lab in directions.EMOTION_LABELS:
    for a in cfg.alphas:
        run.run_turn1(cfg, model, tok, qdf, t0, LIDS, f"add_{lab}_a{a}", ["neutral"],
                      mode="add", dir_for=lambda c, d=dirs[lab]: d, alpha=a, layers=LAYERS)
"""))
cells.append(md("## Phase 5 · Analysis → `results/tables/*.csv`, `results/figures/*.png`"))
cells.append(code("""
t1 = run.load_turn1(cfg)
lens = run.load_lens(cfg)
print(t1.groupby("spec").size())
tables = analysis.write_all(cfg, t0, t1, lens, val, cfg.layer_band, NL + 1)
for name, df in tables.items():
    print("\\n==", name); display(df.round(3))
"""))
cells.append(code("""
from IPython.display import Image
for f in sorted(os.listdir(cfg.path("figures", "x")[:-2])):
    print(f); display(Image(cfg.path("figures", f)))
"""))
cells.append(md("""
**Check before writing:** in `addition.csv`, discard any α where `mean_mass1 < 0.9` (the model has stopped answering coherently).
In `mediation.csv`, the headline is `effect_emo_vs_rand` with its CI for each pressure condition (kind = bad). The `kind = good` rows show whether ablation also blocks *legitimate* updating.
"""))
cells.append(code("""
# Phase 6 · Download everything
!cd /kaggle/working && zip -qr results.zip results && ls -lh results.zip
"""))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(ROOT, "notebooks", "main.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, len(cells), "cells")
