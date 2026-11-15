"""Resumable experiment loops. Outputs are parquet part files under cfg.out_dir."""
from __future__ import annotations

import glob
import os
from typing import Callable

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .hooks import Intervention, logit_lens
from .prompts import LETTERS, build_turn0, build_turn1, pick_target
from .readout import forward_last, letter_probs


# ---------------------------------------------------------------- io helpers
def load_parts(directory: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(directory, "part_*.parquet")))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


class PartWriter:
    def __init__(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        self.dir = directory
        self.k = len(glob.glob(os.path.join(directory, "part_*.parquet")))
        self.buf: list[dict] = []

    def add(self, rows: list[dict]):
        self.buf.extend(rows)

    def flush(self):
        if self.buf:
            pd.DataFrame(self.buf).to_parquet(os.path.join(self.dir, f"part_{self.k:05d}.parquet"))
            self.k += 1
            self.buf = []


def _lens_rows(lens_probs, qids, conds, answer_idx, target_idx, turn):
    rows = []
    lp = lens_probs.numpy()
    for b in range(lp.shape[0]):
        for layer in range(lp.shape[1]):
            rows.append({"qid": int(qids[b]), "condition": conds[b], "turn": turn, "layer": layer,
                         "p_correct": float(lp[b, layer, answer_idx[b]]),
                         "p_target": float(lp[b, layer, target_idx[b]]) if target_idx[b] >= 0 else np.nan})
    return rows


# ---------------------------------------------------------------- turn 0
def run_turn0(cfg, model, tok, qdf: pd.DataFrame, letter_ids, splits: dict[int, str]) -> pd.DataFrame:
    """Baseline answers for all questions (+ turn-0 logit lens). Cached to turn0.parquet."""
    out_path = cfg.path("turn0.parquet")
    if os.path.exists(out_path):
        t0 = pd.read_parquet(out_path)
        if set(qdf["qid"]) <= set(t0["qid"]):
            return t0
    lens_writer = PartWriter(cfg.path("lens", "turn0", "x")[:-2])
    rows = []
    recs = qdf.to_dict("records")
    recs.sort(key=lambda r: len(r["question"]) + sum(len(c) for c in r["choices"]))
    for i in tqdm(range(0, len(recs), cfg.batch_size), desc="turn0"):
        batch = recs[i:i + cfg.batch_size]
        prompts = [build_turn0(tok, q) for q in batch]
        logits, hidden = forward_last(model, tok, prompts, want_hidden=True)
        probs, mass = letter_probs(logits, letter_ids)
        probs, mass = probs.cpu().numpy(), mass.cpu().numpy()
        lens = logit_lens(model, hidden, letter_ids)
        tgt = []
        for b, q in enumerate(batch):
            pred0 = int(probs[b].argmax())
            target_idx, kind = pick_target(q["qid"], q["answer_idx"], pred0, cfg.seed)
            tgt.append(target_idx)
            rows.append({"qid": q["qid"], "subject": q["subject"], "answer_idx": int(q["answer_idx"]),
                         **{f"p0_{L}": float(probs[b, j]) for j, L in enumerate(LETTERS)},
                         "pred0": pred0, "correct0": pred0 == q["answer_idx"],
                         "mass0": float(mass[b]), "split": splits[q["qid"]],
                         "target_idx": target_idx, "kind": kind})
        lens_writer.add(_lens_rows(lens, [q["qid"] for q in batch], ["turn0"] * len(batch),
                                   [q["answer_idx"] for q in batch], tgt, turn=0))
    lens_writer.flush()
    t0 = pd.DataFrame(rows).sort_values("qid").reset_index(drop=True)
    t0.to_parquet(out_path)
    return t0


# ---------------------------------------------------------------- turn 1
def run_turn1(cfg, model, tok, qdf: pd.DataFrame, t0: pd.DataFrame, letter_ids,
              spec: str, conditions: list[str],
              mode: str | None = None, dir_for: Callable[[str], torch.Tensor | None] | None = None,
              alpha: float = 0.0, layers: list[int] | None = None, seed: int = -1,
              qids: list[int] | None = None, want_hidden: bool = False,
              proj_dirs: dict[str, torch.Tensor] | None = None, proj_index: int | None = None,
              flush_every: int = 25) -> None:
    """One pushback turn for every (qid, condition) under one intervention spec.

    spec: unique name, e.g. 'none', 'emo_ablate', 'rand_ablate_s3', 'add_fear_a0.05'.
    mode/dir_for: if set, apply Intervention(mode, dir_for(condition)) on `layers`;
                  conditions where dir_for returns None are skipped.
    want_hidden: also write logit lens rows and emotion projections (spec 'none').
    """
    out_dir = cfg.path("turn1", spec, "x")[:-2]
    done = load_parts(out_dir)
    done_keys = set(zip(done["qid"], done["condition"])) if len(done) else set()
    writer = PartWriter(out_dir)
    lens_writer = PartWriter(cfg.path("lens", spec, "x")[:-2]) if want_hidden else None

    qmap = {r["qid"]: r for r in qdf.to_dict("records")}
    t0map = {r["qid"]: r for r in t0.to_dict("records")}
    use_qids = sorted(qids if qids is not None else t0["qid"].tolist())

    n_batches = 0
    for cond in conditions:
        direction = None
        if mode is not None:
            direction = dir_for(cond)
            if direction is None:
                continue
        todo = [q for q in use_qids if (q, cond) not in done_keys]
        if not todo:
            continue
        items = []
        for q in todo:
            r0 = t0map[q]
            para = q % 3
            prompt = build_turn1(tok, qmap[q], r0["pred0"], cond, r0["target_idx"], para)
            items.append((q, para, prompt))
        items.sort(key=lambda x: len(x[2]))

        ctx = (Intervention(model, layers, mode, direction, alpha) if mode is not None
               else _NullCtx())
        with ctx:
            for i in tqdm(range(0, len(items), cfg.batch_size), desc=f"{spec}/{cond}", leave=False):
                batch = items[i:i + cfg.batch_size]
                logits, hidden = forward_last(model, tok, [b[2] for b in batch], want_hidden=want_hidden)
                probs, mass = letter_probs(logits, letter_ids)
                probs, mass = probs.cpu().numpy(), mass.cpu().numpy()
                rows = []
                for b, (q, para, _) in enumerate(batch):
                    r0 = t0map[q]
                    pred1 = int(probs[b].argmax())
                    row = {"qid": q, "condition": cond, "para_id": para, "spec": spec,
                           "seed": seed, "alpha": alpha,
                           **{f"p1_{L}": float(probs[b, j]) for j, L in enumerate(LETTERS)},
                           "pred1": pred1, "mass1": float(mass[b]),
                           "answer_idx": r0["answer_idx"], "pred0": r0["pred0"],
                           "target_idx": r0["target_idx"], "kind": r0["kind"],
                           "changed": pred1 != r0["pred0"],
                           "moved_to_target": pred1 == r0["target_idx"],
                           # abandoned a correct answer (works for no-suggestion control too)
                           "bad_flip": r0["kind"] == "bad" and pred1 != r0["answer_idx"],
                           "good_update": r0["kind"] == "good" and pred1 == r0["answer_idx"]}
                    if want_hidden and proj_dirs is not None and proj_index is not None:
                        h = hidden[b, proj_index].cpu()
                        for lab, d in proj_dirs.items():
                            row[f"proj_{lab}"] = float(h @ d[proj_index])
                    rows.append(row)
                writer.add(rows)
                if want_hidden:
                    lens = logit_lens(model, hidden, letter_ids)
                    lens_writer.add(_lens_rows(
                        lens, [b[0] for b in batch], [cond] * len(batch),
                        [t0map[b[0]]["answer_idx"] for b in batch],
                        [t0map[b[0]]["target_idx"] if cond != "control_nosuggest" else -1
                         for b in batch], turn=1))
                n_batches += 1
                if n_batches % flush_every == 0:
                    writer.flush()
                    if lens_writer:
                        lens_writer.flush()
    writer.flush()
    if lens_writer:
        lens_writer.flush()


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def load_turn1(cfg) -> pd.DataFrame:
    dirs = sorted(glob.glob(os.path.join(cfg.out_dir, "turn1", "*")))
    dfs = [load_parts(d) for d in dirs]
    dfs = [d for d in dfs if len(d)]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_lens(cfg) -> pd.DataFrame:
    dirs = sorted(glob.glob(os.path.join(cfg.out_dir, "lens", "*")))
    dfs = [load_parts(d) for d in dirs]
    dfs = [d for d in dfs if len(d)]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
