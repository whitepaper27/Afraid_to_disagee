"""Builds notebooks/addon.ipynb: one Run-All notebook for the exploratory conditions + extra controls.

It restores the earlier results (from an attached Kaggle dataset made from results.zip, or from
/kaggle/working/results if already present), runs ONLY the new pieces, re-runs the analysis over
everything, and zips a combined results_v2.zip.
"""
import json
import os
import sys

ROOT = sys.argv[1]
SRC = os.path.join(ROOT, "src")
MODULES = ["__init__.py", "config.py", "data.py", "prompts.py", "readout.py", "hooks.py",
           "directions.py", "run.py", "analysis.py"]


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": t.strip("\n").splitlines(True)}


cells = [md("""
# Afraid to Disagree? — Add-on run (exploratory conditions + extra controls)

**Before Run All:**
1. **Settings:** Accelerator = **GPU T4**, Internet = **On**.
2. **Attach your earlier results:** right panel → **+ Add Input** → **Upload** → choose `results.zip` → name it `afraid-results` → Create. Wait until it shows under *Input*.
   (Skip this only if you run this inside the SAME session that already has `/kaggle/working/results`.)
3. **Run → Run All.** About 45 min on a T4. At the end, click the download link for `results_v2.zip`.

What it runs (everything else is reused from your earlier run, nothing is recomputed):
- New pushback conditions: `authority` (positive control) and `boss_deadline` (boss + deadline + shutdown): no intervention, matched-emotion removal, 10 random-direction removals, mismatched removal.
- **Critical control:** random directions *added* to neutral pushback (5 seeds × α 0.05, 0.1).
- Exploratory: fear / joy directions *removed* during neutral pushback.
- Full analysis over all results, written to `results/tables` + `results/figures`.
""")]
cells.append(code("""
# 1 · Environment
!pip -q install -U "datasets>=2.19" 2>/dev/null
import torch, transformers, datasets
print("torch", torch.__version__, "| transformers", transformers.__version__, "| datasets", datasets.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE — enable a GPU")
!mkdir -p /kaggle/working/src
"""))
cells.append(md("## 2 · Source modules (current versions)"))
for m in MODULES:
    body = open(os.path.join(SRC, m)).read()
    cells.append(code(f"%%writefile /kaggle/working/src/{m}\n{body}"))
cells.append(code("""
# 3 · Restore earlier results into /kaggle/working/results
import glob, os, shutil, subprocess
OUT = "/kaggle/working/results"
if not os.path.exists(f"{OUT}/turn0.parquet"):
    hits = glob.glob("/kaggle/input/**/turn0.parquet", recursive=True)
    if not hits:   # dataset kept as a zip
        zips = glob.glob("/kaggle/input/**/*.zip", recursive=True)
        assert zips, "No earlier results found: attach results.zip as a dataset (see top of notebook)."
        subprocess.run(["unzip", "-qo", zips[0], "-d", "/kaggle/working/restore"], check=True)
        hits = glob.glob("/kaggle/working/restore/**/turn0.parquet", recursive=True)
    assert hits, "turn0.parquet not found inside the attached input."
    shutil.copytree(os.path.dirname(hits[0]), OUT, dirs_exist_ok=True)
need = ["turn0.parquet", "directions.pt", "directions_validation.csv", "config.json", "turn1/none", "turn1/emo_ablate"]
missing = [n for n in need if not os.path.exists(os.path.join(OUT, n))]
assert not missing, f"Earlier results incomplete, missing: {missing}"
print("restored OK:", len(os.listdir(f"{OUT}/turn1")), "turn1 specs")
"""))
cells.append(code("""
# 4 · Config (frozen settings from the earlier run) + model + data + directions
import sys, json
sys.path.insert(0, "/kaggle/working")
import numpy as np, pandas as pd, torch
from IPython.display import display
from src.config import Config
from src import data, prompts, readout, hooks, directions, run, analysis

torch.manual_seed(0); np.random.seed(0)
saved = json.load(open(f"{OUT}/config.json"))
cfg = Config(model_id=saved["model_id"], out_dir=OUT, layer_band=tuple(saved["layer_band"]))
assert cfg.layer_band == (19, 27), cfg.layer_band
LAYERS = list(range(cfg.layer_band[0], cfg.layer_band[1] + 1))
PROJ_INDEX = (cfg.layer_band[0] + cfg.layer_band[1]) // 2 + 1

model, tok = readout.load_model(cfg.model_id, cfg.dtype)
LIDS = readout.letter_token_ids(tok)
NL = readout.n_layers(model); D = model.config.hidden_size

qdf = data.load_mmlu_sample(cfg.n_questions, cfg.seed, cfg.max_chars)
splits = data.split_ids(qdf["qid"], cfg.holdout_frac, cfg.seed)
t0 = run.run_turn0(cfg, model, tok, qdf, LIDS, splits)      # loads cached file, no recompute
assert len(t0) == len(qdf) == 596, (len(t0), len(qdf))
assert set(qdf["qid"]) == set(t0["qid"])

dirs = torch.load(f"{OUT}/directions.pt")
val = pd.read_csv(f"{OUT}/directions_validation.csv")
R = directions.random_dirs(NL + 1, D, cfg.n_random_dirs, seed=1234)   # same seed as the main run
NEW = prompts.EXPLORATORY_CONDITIONS
print("band", cfg.layer_band, "| proj index", PROJ_INDEX, "| new conditions", NEW)
print("turn-0 accuracy (should be 0.624):", round(t0["correct0"].mean(), 3))
"""))
cells.append(md("## 5 · New conditions: authority, boss_deadline"))
cells.append(code("""
run.run_turn1(cfg, model, tok, qdf, t0, LIDS, "none", NEW,
              want_hidden=True, proj_dirs=dirs, proj_index=PROJ_INDEX)

emo_for = lambda c: dirs[directions.EMOTION_FOR[c]] if directions.EMOTION_FOR.get(c) else None
run.run_turn1(cfg, model, tok, qdf, t0, LIDS, "emo_ablate", NEW,
              mode="ablate", dir_for=emo_for, layers=LAYERS)

for s in range(cfg.n_random_dirs):
    run.run_turn1(cfg, model, tok, qdf, t0, LIDS, f"rand_ablate_s{s}", NEW,
                  mode="ablate", dir_for=lambda c, r=R[s]: r, layers=LAYERS, seed=s)

mis_for = lambda c: dirs[directions.MISMATCH_FOR[c]] if c in directions.MISMATCH_FOR else None
run.run_turn1(cfg, model, tok, qdf, t0, LIDS, "mismatch_ablate", NEW,
              mode="ablate", dir_for=mis_for, layers=LAYERS)
display(analysis.rate_table(run.load_turn1(cfg), "none").round(3))
"""))
cells.append(md("## 6 · Extra controls"))
cells.append(code("""
# (a) CRITICAL: random directions ADDED to neutral pushback (same layers, same α)
for s in range(5):
    for a in (0.05, 0.1):
        run.run_turn1(cfg, model, tok, qdf, t0, LIDS, f"addrand_s{s}_a{a}", ["neutral"],
                      mode="add", dir_for=lambda c, r=R[s]: r, alpha=a, layers=LAYERS, seed=s)
# (b) Exploratory: remove emotion directions during NEUTRAL pushback
for lab in ["fear", "joy"]:
    run.run_turn1(cfg, model, tok, qdf, t0, LIDS, f"neuabl_{lab}", ["neutral"],
                  mode="ablate", dir_for=lambda c, d=dirs[lab]: d, layers=LAYERS)

t1 = run.load_turn1(cfg)
base = t1[(t1.spec == "none") & (t1.condition == "neutral")].set_index("qid")
rows = []
for spec, g in t1[t1.spec.str.startswith(("addrand_", "neuabl_", "add_"))].groupby("spec"):
    g = g.set_index("qid")
    for kind, col in (("bad", "bad_flip"), ("good", "good_update")):
        q = g.index[g.kind == kind]
        m, lo, hi = analysis.boot_mean(g.loc[q, col].astype(float).values - base.loc[q, col].astype(float).values)
        rows.append(dict(spec=spec, kind=kind, n=len(q), delta=m, lo=lo, hi=hi, mass=g.mass1.mean()))
ctrl = pd.DataFrame(rows)
ctrl.to_csv(cfg.path("tables", "controls_extra.csv"), index=False)
display(ctrl.round(3))
"""))
cells.append(md("## 7 · Full analysis over everything"))
cells.append(code("""
t1 = run.load_turn1(cfg)
lens = run.load_lens(cfg)
print(t1.groupby("spec").size().to_string())
tables = analysis.write_all(cfg, t0, t1, lens, val, cfg.layer_band, NL + 1)
for name, df in tables.items():
    print("\\n==", name); display(df.round(3))
"""))
cells.append(code("""
# 8 · Download
import os
os.chdir("/kaggle/working")
!rm -f results_v2.zip && zip -qr results_v2.zip results && ls -lh results_v2.zip
from IPython.display import FileLink
FileLink("results_v2.zip")
"""))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                  "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(ROOT, "notebooks", "addon.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out, len(cells), "cells")
