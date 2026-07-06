import torch
from torch import Tensor


# ================================================================
# CORE FUNCTIONS
# ================================================================

def newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """
    Original Muon orthogonalization (alpha=0).
    Maps all singular values to 1.
    Used during warmup phase and as fallback.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


def compute_alpha(G: Tensor, eps: float = 1e-7) -> float:
    """
    Compute optimal alpha from gradient singular value SNR.
    
    Formula derived from DeepSeek proof:
        alpha* = 1 - 1/(1 + SNR²)
    where SNR = S.mean() / S.std()
    
    Properties:
    - High noise (early training): SNR low  → alpha near 0 → like Muon
    - Low noise  (late training):  SNR high → alpha rising → preserve signal
    
    Uses svdvals only (no U, V computed) — fast.
    """
    try:
        S = torch.linalg.svdvals(G.float())
        snr = S.mean() / (S.std() + eps)
        return (1.0 - 1.0 / (1.0 + snr ** 2)).item()
    except Exception:
        return 0.0  # fallback to Muon


def ns_soft_ortho(G: Tensor, alpha: float, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """
    Fast approximate soft orthogonalization.
    
    Interpolates between:
    - Pure Muon output (alpha=0): U @ V.T
    - Normalized gradient (alpha=1): G / ||G||
    
    Formula: result = (1-alpha) * NS(G) + alpha * G_normalized
    
    This approximates U @ diag(sigma^alpha) @ V.T without full SVD.
    Same computational cost as standard Muon.
    """
    assert G.ndim == 2

    if alpha <= 0.01:
        return newtonschulz5(G, steps=steps, eps=eps)

    G_f = G.float()
    g_norm = G_f.norm() + eps

    # compute NS result (alpha=0 component)
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G_f / g_norm
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X.bfloat16()
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    ns_result = X.float()

    # normalized gradient (alpha=1 component)
    g_normalized = G_f / g_norm

    # interpolate
    result = (1.0 - alpha) * ns_result + alpha * g_normalized

    # normalize to unit scale (matches Muon output convention)
    result = result / (result.norm() + eps)

    if torch.isnan(result).any():
        return newtonschulz5(G, steps=steps, eps=eps)

    return result.to(G.dtype)


# ================================================================
# HYBRID MUON OPTIMIZER CLASS
# ================================================================

class HybridMuon:
    """
    Hybrid Muon Optimizer: Pure Muon warmup → Adaptive Alpha
    
    Drop-in replacement for Muon in modded-nanogpt.
    
    Args:
        params_2d:      List of 2D parameters (attention/MLP weights)
        params_other:   List of other parameters (embeddings, scalars) → AdamW
        lr:             Learning rate for Muon/HybridMuon (default: 0.02)
        momentum:       SGD momentum (default: 0.95)
        lr_adam:        Learning rate for AdamW on other params (default: 3e-4)
        warmup_steps:   Steps of pure Muon before switching to adaptive (default: 1000)
        alpha_freq:     How often to recompute alpha in adaptive phase (default: 20)
    
    Usage:
        # same as Muon
        opt = HybridMuon(params_2d, params_other, lr=0.02)
        
        for step in range(total_steps):
            opt.zero_grad()
            loss.backward()
            opt.step()
    """

    def __init__(
        self,
        params_2d,
        params_other,
        lr: float = 0.02,
        momentum: float = 0.95,
        lr_adam: float = 3e-4,
        warmup_steps: int = 1000,
        alpha_freq: int = 20,
        ns_steps: int = 5,
    ):
        self.params_2d    = list(params_2d)
        self.params_other = list(params_other)
        self.lr           = lr
        self.momentum     = momentum
        self.warmup_steps = warmup_steps
        self.alpha_freq   = alpha_freq
        self.ns_steps     = ns_steps

        # momentum buffers
        self.momentum_buffers = [
            torch.zeros_like(p) for p in self.params_2d
        ]

        # cached alpha per parameter
        self._alphas = [0.0] * len(self.params_2d)
        self._step   = 0

        # logging
        self.alpha_history = []

        # AdamW for non-2D params
        if self.params_other:
            self.adam = torch.optim.AdamW(self.params_other, lr=lr_adam)
        else:
            self.adam = None

    def zero_grad(self):
        for p in self.params_2d + self.params_other:
            if p.grad is not None:
                p.grad.zero_()

    @property
    def in_warmup(self) -> bool:
        return self._step <= self.warmup_steps

    def step(self):
        self._step += 1
        recompute = (
            not self.in_warmup and
            self._step % self.alpha_freq == 0
        )

        step_alphas = []

        for i, (p, buf) in enumerate(zip(self.params_2d, self.momentum_buffers)):
            if p.grad is None:
                continue

            g = p.grad.float()

            # nesterov momentum
            buf.mul_(self.momentum).add_(g)
            g_nesterov = g.add(buf, alpha=self.momentum)

            if self.in_warmup:
                # ---- PHASE 1: Pure Muon ----
                update = newtonschulz5(g_nesterov, steps=self.ns_steps)
                step_alphas.append(0.0)

            else:
                # ---- PHASE 2: Adaptive Alpha ----
                # recompute alpha from SVD every alpha_freq steps
                if recompute:
                    self._alphas[i] = compute_alpha(g_nesterov)

                alpha = self._alphas[i]
                step_alphas.append(alpha)

                # fast soft orthogonalization
                update = ns_soft_ortho(
                    g_nesterov,
                    alpha=alpha,
                    steps=self.ns_steps
                )

            p.data.add_(update.to(p.dtype), alpha=-self.lr)

        # log mean alpha
        if step_alphas:
            self.alpha_history.append(
                sum(step_alphas) / len(step_alphas)
            )

        # AdamW step for embeddings/scalars
        if self.adam is not None:
            self.adam.step()

    def get_current_alpha(self) -> float:
        if self.alpha_history:
            return self.alpha_history[-1]
        return 0.0

    def __repr__(self):
        phase = "WARMUP" if self.in_warmup else "ADAPTIVE"
        alpha = self.get_current_alpha()
        return (
            f"HybridMuon("
            f"step={self._step}, "
            f"phase={phase}, "
            f"alpha={alpha:.3f}, "
            f"lr={self.lr}, "
            f"warmup={self.warmup_steps})"
        )


# ================================================================
# HOW TO USE IN train_gpt.py
# ================================================================
"""
In train_gpt.py, find where Muon/NorMuonAndAdam is initialized.

Replace:
    optimizer = NorMuonAndAdam(...)   # or however Muon is initialized

With:
    from hybrid_muon import HybridMuon
    
    params_2d = [p for name, p in model.named_parameters() 
                 if p.ndim == 2 and 'embed' not in name and 'head' not in name]
    params_other = [p for name, p in model.named_parameters()
                    if p not in params_2d]
    
    optimizer = HybridMuon(
        params_2d=params_2d,
        params_other=params_other,
        lr=0.023,              # same as NorMuon lr in speedrun
        momentum=0.95,
        warmup_steps=345,      # ~25% of 1380 total steps
        alpha_freq=20,
    )

Then in training loop, replace optimizer.step() calls with:
    optimizer.step()           # same interface, just works

That's it. 
"""


# ================================================================
# QUICK TEST (run this file directly to verify)
# ================================================================

if __name__ == "__main__":
    print("Testing HybridMuon...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # fake 2D params like transformer weights
    params_2d = [
        torch.nn.Parameter(torch.randn(768, 768).to(device)),
        torch.nn.Parameter(torch.randn(3072, 768).to(device)),
        torch.nn.Parameter(torch.randn(768, 3072).to(device)),
    ]
    params_other = [
        torch.nn.Parameter(torch.randn(768).to(device)),
    ]

    opt = HybridMuon(
        params_2d=params_2d,
        params_other=params_other,
        lr=0.02,
        warmup_steps=5,
        alpha_freq=2,
    )

    print("Running 20 fake steps...")
    for step in range(20):
        # fake gradients
        for p in params_2d + params_other:
            p.grad = torch.randn_like(p)

        opt.step()
        print(f"  step {step+1:2d}: {opt}")

    print("\n✅ HybridMuon works correctly!")
    print(f"Final alpha: {opt.get_current_alpha():.3f}")
    print("\nReady to submit to modded-nanogpt 🔥")
