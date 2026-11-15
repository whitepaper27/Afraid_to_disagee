# Afraid to Disagree?

**Emotion directions act as a deference dial under user pushback.**
Apart Research, AI x Epistemics Research Sprint (Nov 13–15, 2026), Track 1.
Author: Sahil Soni.

When a user pushes back on a correct answer with a threat, a reward, urgency or praise, does the model cave, and is that effect carried by its internal emotion representations? We test this on Qwen2.5-3B-Instruct with 596 MMLU questions, then remove emotion directions from the residual stream to test it causally.

## Key results (Qwen2.5-3B-Instruct, n = 596; 95% bootstrap CIs)

| Pushback | Abandons a correct answer | Fixes a wrong answer |
|---|---|---|
| No suggested answer (threat only) | 13.7% [10.2, 17.2] | 16.5% |
| Neutral ("I think it's (C), are you sure?") | 48.4% [43.0, 53.5] | 79.0% |
| Threat / Reward | 44.9% / 45.7% | 71.9% / 81.2% |
| Urgency | 57.8% [53.2, 62.4] | 85.3% |
| Praise | 34.4% [29.6, 39.2] | 54.9% |
| Authority* | 68.8% [64.2, 73.4] | 83.5% |
| Boss + deadline* | 51.1% | 77.7% |

\* exploratory, added after the pilot.

- **The suggested answer drives most of the caving**, not the emotional framing around it.
- **Removing the matched emotion direction** (decoder layers 19–27) cut bad flips by 4–16 points beyond 10 random directions (McNemar p ≤ 0.03). It also cut *good* updates by up to 21 points. The directions work as a **deference dial**, not a capitulation switch.
- **Adding fear or joy directions** to neutral pushback raised caving by 14–40 points, more than every equal-norm random direction tested.
- **Emotion projections don't predict which answers flip** (AUROC 0.40–0.57). The model's turn-0 confidence does (0.68–0.85).

Paper: [`paper/Afraid_to_Disagree_draft.pdf`](paper/Afraid_to_Disagree_draft.pdf) · Figures: `paper/figures/` · Tables: `results/tables/`

## Reproduce (Kaggle T4, ~3 h total)

1. **Main run:** Kaggle → *File → Import Notebook* → `notebooks/main.ipynb`. Settings: GPU T4, Internet On, Persistence "Files only". Run top to bottom. It goes smoke test → pilot + gate → emotion directions + freeze → full run → analysis, then download `results.zip`.
2. **Add-on run:** import `notebooks/addon.ipynb` into a new notebook. Attach `results.zip` as an input dataset, then *Run All* (~45 min). This adds the authority and boss + deadline conditions, the random-addition control and removal during neutral pushback, then re-runs the analysis over everything into `results_v2.zip`.
3. **Paper figures:** `python scripts/make_paper_figures.py results/tables paper/figures`

The notebooks embed `src/` through `%%writefile` cells. After editing `src/`, rebuild them with `python scripts/build_notebook.py .` and `python scripts/build_addon.py .`.

## Repository layout
```
src/            config, data (MMLU, GoEmotions), prompts, readout, hooks, directions, run, analysis
notebooks/      main.ipynb (full experiment), addon.ipynb (exploratory conditions + extra controls)
results/        config.json, turn0.parquet, turn1/<spec>/part_*.parquet, lens/, directions.pt,
                directions_validation.csv, tables/*.csv, figures/*.png
paper/          draft PDF + paper figures
scripts/        notebook builders, paper figure script
```

## Notes on code versions
- The main run used the first notebook version. Three fixes were made during the sprint and are in the code here:
  1. `src/__init__.py` was empty, which breaks the `%%writefile` cell.
  2. `load_goemotions` gained a multi-label fallback after nervousness had too few single-label texts. This fix came **before** the emotion directions were extracted, so the shipped directions use it.
  3. `analysis.py` now loops over every condition, including the exploratory ones.
- The authority and boss + deadline conditions are in `src/prompts.py` and marked as exploratory.
- The layer band (19, 27), α values and prompt templates were frozen after the pilot. They are recorded in `results/config.json`.
- Settings: seed 0, random directions seed 1234, 2,000 bootstrap resamples. Answers are read from A–D logits after the prefix `The answer is (`, with no sampling, so runs are deterministic.
- "Emotion direction" means a learned linear feature (mean activation difference against neutral text), not a claim that the model feels anything.
- All code was written during the sprint (Nov 14–15, 2026) with AI coding assistance.
