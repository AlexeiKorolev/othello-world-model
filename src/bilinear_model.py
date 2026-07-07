"""GPT-1 (minGPT) with bilinear MLPs replacing the GELU feed-forward blocks.

Bilinear MLP (Dooms, Pearce, Rigg, Oramas, Sharkey 2024, arXiv:2410.08417,
ICLR 2025): a Gated Linear Unit with the nonlinear gate removed --

    g(x)   = (W x) * (V x)     # elementwise product of two linear maps
    MLP(x) = P g(x)            # output projection, no bias

Because there is no elementwise activation, the layer is a *bilinear form* in x
and its computation is captured entirely by a third-order weight tensor, which
can be eigendecomposed for weight-based mechanistic interpretability. We keep
everything else identical to the baseline Othello-GPT (8L/8H/512d, LayerNorm,
residual stream, causal attention) and swap only the MLP.

Expansion factor (d_hidden = expansion * d_model):
  * expansion=4      -> paper-faithful; MLP has 3 matrices = 12 d^2 params,
                        ~1.5x the GELU baseline's 8 d^2 (and ~1.5x MLP FLOPs).
  * expansion=8/3    -> parameter- AND FLOP-matched to the GELU baseline
                        (3 * 8/3 d^2 = 8 d^2). Cleanest controlled comparison.

No biases, gpt2-style init (std 0.02), matching the paper.
"""
import torch
import torch.nn as nn


class BilinearMLP(nn.Module):
    def __init__(self, n_embd, expansion=4.0, resid_pdrop=0.1, bias=False):
        super().__init__()
        hidden = int(round(expansion * n_embd))
        self.hidden = hidden
        self.w = nn.Linear(n_embd, hidden, bias=bias)     # left linear map
        self.v = nn.Linear(n_embd, hidden, bias=bias)     # right (gate) linear map
        self.proj = nn.Linear(hidden, n_embd, bias=bias)  # output projection
        self.drop = nn.Dropout(resid_pdrop)

    def forward(self, x):
        return self.drop(self.proj(self.w(x) * self.v(x)))


def make_bilinear_gpt(mconf, expansion=4.0):
    """Build a minGPT GPT, then replace every block's GELU MLP with a
    BilinearMLP. All other components (attention, LayerNorm, embeddings, head)
    and the GPT forward / configure_optimizers logic are untouched, so the model
    is a drop-in for the training / probing / eval pipeline."""
    from mingpt.model import GPT
    model = GPT(mconf)
    resid_pdrop = getattr(mconf, "resid_pdrop", 0.1)
    for blk in model.blocks:
        blk.mlp = BilinearMLP(mconf.n_embd, expansion, resid_pdrop)
        blk.mlp.apply(model._init_weights)   # gpt2-style init for the new params
    return model
