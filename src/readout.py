"""Model loading and A-D logit readout."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompts import LETTERS


def load_model(model_id: str, dtype: str = "float16"):
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=getattr(torch, dtype), device_map={"": 0})
    model.eval()
    return model, tok


def letter_token_ids(tok) -> list[int]:
    ids = []
    for L in LETTERS:
        t = tok.encode(L, add_special_tokens=False)
        assert len(t) == 1, f"Letter {L} is not a single token: {t}"
        ids.append(t[0])
    return ids


def n_layers(model) -> int:
    return len(model.model.layers)


@torch.no_grad()
def forward_last(model, tok, prompts: list[str], want_hidden: bool = False):
    """Returns (last-position logits [B,V] float32, last-token hidden [B, L+1, d] or None).

    Left padding + explicit position_ids so padded rows get correct positions.
    """
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    input_ids = enc["input_ids"].to(model.device)
    attn = enc["attention_mask"].to(model.device)
    pos = (attn.long().cumsum(-1) - 1).clamp(min=0)
    out = model(input_ids=input_ids, attention_mask=attn, position_ids=pos,
                output_hidden_states=want_hidden, use_cache=False)
    logits = out.logits[:, -1, :].float()
    hidden = None
    if want_hidden:
        hidden = torch.stack([h[:, -1, :] for h in out.hidden_states], dim=1).float()
    return logits, hidden


def letter_probs(logits: torch.Tensor, letter_ids: list[int]):
    """probs over A-D [B,4] and total full-vocab mass on A-D [B] (coherence)."""
    mass = logits.softmax(-1)[:, letter_ids].sum(-1)
    probs = logits[:, letter_ids].softmax(-1)
    return probs, mass
