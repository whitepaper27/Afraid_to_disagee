"""Run configuration. Every run writes its config to results/config.json."""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass


@dataclass
class Config:
    model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    dtype: str = "float16"
    n_questions: int = 600          # full sample size (pilot uses qid < pilot_n)
    pilot_n: int = 100
    max_chars: int = 1500           # drop very long MMLU items (padding cost)
    holdout_frac: float = 0.2
    seed: int = 0
    batch_size: int = 8
    n_emotion_texts: int = 400      # per GoEmotions label (train); 100 held out
    layer_band: tuple | None = None # (start, end) inclusive, frozen after pilot
    alphas: tuple = (0.02, 0.05, 0.1)
    n_random_dirs: int = 10
    out_dir: str = "/kaggle/working/results"

    def path(self, *parts: str) -> str:
        p = os.path.join(self.out_dir, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def save(self) -> None:
        d = dataclasses.asdict(self)
        with open(self.path("config.json"), "w") as f:
            json.dump(d, f, indent=2, default=list)

    def frozen(self, **changes) -> "Config":
        return dataclasses.replace(self, **changes)
