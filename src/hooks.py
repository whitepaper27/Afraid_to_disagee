"""Activation interventions (ablate / add) and logit lens.

Layer convention: hidden_states index i+1 == output of decoder layer i.
Directions are stored as Tensor[L+1, d]; the hook on layer i uses row i+1.
"""
from __future__ import annotations

import torch


class Intervention:
    """Context manager. mode='ablate': h <- h - (h.d)d ; mode='add': h <- h + alpha*||h||*d.

    Applied at every token position on the given decoder layers.
    """

    def __init__(self, model, layers: list[int], mode: str, dirs: torch.Tensor,
                 alpha: float = 0.0):
        assert mode in ("ablate", "add")
        self.model, self.layers, self.mode, self.alpha = model, list(layers), mode, alpha
        d = dirs.float()
        self.dirs = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.handles = []

    def _make_hook(self, layer_idx: int):
        d = self.dirs[layer_idx + 1].to(self.model.device)

        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            h32 = h.float()
            if self.mode == "ablate":
                h32 = h32 - (h32 @ d).unsqueeze(-1) * d
            else:
                h32 = h32 + self.alpha * h32.norm(dim=-1, keepdim=True) * d
            h_new = h32.to(h.dtype)
            if isinstance(output, tuple):
                return (h_new,) + tuple(output[1:])
            return h_new

        return hook

    def __enter__(self):
        for i in self.layers:
            self.handles.append(self.model.model.layers[i].register_forward_hook(self._make_hook(i)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def logit_lens(model, hidden: torch.Tensor, letter_ids: list[int]) -> torch.Tensor:
    """hidden [B, L+1, d] -> probs over A-D per layer [B, L+1, 4].

    HF's final hidden state already has the final norm applied, so the last layer
    is projected directly; earlier layers go through model.model.norm first.
    """
    W = model.lm_head.weight[letter_ids].float()             # [4, d]
    h = hidden.to(model.device)
    normed = model.model.norm(h[:, :-1, :].to(model.dtype)).float()
    normed = torch.cat([normed, h[:, -1:, :].float()], dim=1)
    return (normed @ W.T).softmax(-1).cpu()
