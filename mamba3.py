# mamba3.py — Mamba-3 implementation based on Mamba-2 structure
# Implements key changes from "Mamba-3: Improved Sequence Modeling using State Space Principles"
# (Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu — ICLR 2026 Oral)
#
# Changes over Mamba-2:
#   1. Trapezoidal discretization (replaces Euler; removes need for short conv1d)
#   2. Complex-valued state update via data-dependent RoPE on B and C
#   3. MIMO (Multi-Input Multi-Output) with configurable rank r
#   4. BC bias (trainable, init=1, enables simultaneous LTI + data-dependent SSM)
#   5. QK-style normalization on B and C
#   6. No short convolution (redundant with trapezoidal + BC bias)

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from huggingface_hub import PyTorchModelHubMixin


def _apply_rope(x, cos, sin):
    """Apply rotary position embeddings (RoPE) to tensor x.
    
    x:   (..., N)  where N is even — will be split into pairs
    cos: (..., N//2)
    sin: (..., N//2)
    """
    N = x.shape[-1]
    x1 = x[..., : N // 2]
    x2 = x[..., N // 2 :]
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.cat([out1, out2], dim=-1)


def _compute_cumulative_angles(angles):
    """Compute cumulative sum of rotation angles along the sequence dimension.
    
    angles: (batch, seqlen, nheads, d_state//2)
    Returns: (batch, seqlen, nheads, d_state//2)
    """
    return torch.cumsum(angles, dim=1)


class Mamba3(nn.Module, PyTorchModelHubMixin):
    """Mamba-3: SSM layer with trapezoidal discretization, complex state dynamics,
    and optional MIMO formulation.
    """

    def __init__(
        self,
        d_model,
        d_state=128,
        expand=2,
        headdim=64,
        d_ssm=None,
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        # ----- Mamba-3 specific -----
        use_rope=True,            # complex SSM via data-dependent RoPE
        use_trapezoidal=True,     # trapezoidal discretization gate
        mimo_rank=1,              # r=1 is SISO (Mamba-2 equivalent); r>1 is MIMO
        bc_bias=True,             # trainable BC bias (init=1)
        qk_norm=True,             # QK-style normalization on B, C
        # ----------------------------
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # Mamba-3 specific flags
        self.use_rope = use_rope
        self.use_trapezoidal = use_trapezoidal
        self.mimo_rank = mimo_rank
        self.bc_bias = bc_bias
        self.qk_norm = qk_norm

        # ── Input projection ──
        # Mamba-3 removes conv1d, so we project: [z, x, B, C, dt]
        # For MIMO rank r: x is expanded by factor r for the input side
        # B stays (ngroups * d_state), C stays (ngroups * d_state)
        d_in_proj = (
            self.d_inner                              # z (gate)
            + self.d_ssm * self.mimo_rank             # x (MIMO: r copies)
            + 2 * self.ngroups * self.d_state         # B and C
            + self.nheads                             # dt
        )

        # Trapezoidal gate λ_t = σ(u_t) — one scalar per head
        if self.use_trapezoidal:
            d_in_proj += self.nheads  # lambda_t projection

        # Data-dependent RoPE angles — one angle per (head, d_state//2) pair
        if self.use_rope:
            d_in_proj += self.nheads * (self.d_state // 2)  # θ_t angles

        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(
                self.d_model, d_in_proj * self.world_size, bias=bias,
                process_group=self.process_group,
                sequence_parallel=self.sequence_parallel,
                **factory_kwargs,
            )

        self.act = nn.SiLU()

        # ── No conv1d in Mamba-3 ──
        # (trapezoidal discretization + BC bias make it redundant)

        # ── Initialize log dt bias ──
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # ── A parameter ──
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # ── D "skip" parameter ──
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        # ── BC bias (Mamba-3): trainable, init=1 ──
        if self.bc_bias:
            self.B_bias = nn.Parameter(torch.ones(self.ngroups * self.d_state, **factory_kwargs))
            self.C_bias = nn.Parameter(torch.ones(self.ngroups * self.d_state, **factory_kwargs))
            self.B_bias._no_weight_decay = True
            self.C_bias._no_weight_decay = True

        # ── QK-style normalization on B, C ──
        if self.qk_norm:
            self.B_norm = nn.RMSNorm(self.d_state, eps=1e-5, **factory_kwargs)
            self.C_norm = nn.RMSNorm(self.d_state, eps=1e-5, **factory_kwargs)

        # ── Output norm ──
        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(
                self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                group_size=self.d_ssm // ngroups, **factory_kwargs,
            )

        # ── MIMO output projection ──
        # For MIMO: we get r outputs per head, need to project back to headdim
        if self.mimo_rank > 1:
            self.mimo_out_proj = nn.Linear(
                self.d_ssm * self.mimo_rank, self.d_ssm, bias=False, **factory_kwargs
            )

        # ── Output projection ──
        if self.process_group is None:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(
                self.d_inner * self.world_size, self.d_model, bias=bias,
                process_group=self.process_group,
                sequence_parallel=self.sequence_parallel,
                **factory_kwargs,
            )

    def _split_projection(self, zxbcdt):
        """Split the input projection into components."""
        splits = []
        sizes = [
            self.d_inner,                             # z
            self.d_ssm * self.mimo_rank,              # x (MIMO-expanded)
            self.ngroups * self.d_state,               # B
            self.ngroups * self.d_state,               # C
            self.nheads,                               # dt
        ]
        if self.use_trapezoidal:
            sizes.append(self.nheads)                  # lambda (trapezoidal gate)
        if self.use_rope:
            sizes.append(self.nheads * (self.d_state // 2))  # theta (RoPE angles)

        return torch.split(zxbcdt, sizes, dim=-1)

    def _apply_bc_processing(self, B, C):
        """Apply BC bias and QK-norm to B and C projections."""
        # BC bias: add trainable offset (enables simultaneous LTI + data-dependent)
        if self.bc_bias:
            B = B + self.B_bias
            C = C + self.C_bias

        # QK-style normalization per group
        if self.qk_norm:
            B_shape = B.shape
            B = B.view(*B_shape[:-1], self.ngroups, self.d_state)
            B = self.B_norm(B)
            B = B.view(*B_shape)

            C_shape = C.shape
            C = C.view(*C_shape[:-1], self.ngroups, self.d_state)
            C = self.C_norm(C)
            C = C.view(*C_shape)

        return B, C

    def _apply_data_dependent_rope(self, B, C, theta_raw):
        """Apply data-dependent RoPE rotations to B and C.
        
        This implements the complex-valued SSM via the equivalence:
        complex A with data-dependent angles ≡ real A with RoPE on B, C.
        
        B: (batch, seqlen, ngroups * d_state)
        C: (batch, seqlen, ngroups * d_state)
        theta_raw: (batch, seqlen, nheads * d_state//2)
        """
        batch, seqlen = B.shape[:2]

        # Reshape angles: (B, L, nheads, d_state//2)
        angles = theta_raw.view(batch, seqlen, self.nheads, self.d_state // 2)

        # Compute cumulative angles along sequence
        cum_angles = _compute_cumulative_angles(angles)  # (B, L, nheads, d_state//2)

        cos_vals = torch.cos(cum_angles)
        sin_vals = torch.sin(cum_angles)

        # Reshape B, C for per-group RoPE: (B, L, ngroups, d_state)
        B = B.view(batch, seqlen, self.ngroups, self.d_state)
        C = C.view(batch, seqlen, self.ngroups, self.d_state)

        # Expand cos/sin from nheads to ngroups (nheads = ngroups * heads_per_group)
        heads_per_group = self.nheads // self.ngroups
        # cos/sin: (B, L, nheads, d_state//2) -> average within group for B,C application
        # Since B,C are per-group, we use the group-level angles
        cos_g = cos_vals.view(batch, seqlen, self.ngroups, heads_per_group, self.d_state // 2)
        sin_g = sin_vals.view(batch, seqlen, self.ngroups, heads_per_group, self.d_state // 2)
        cos_g = cos_g.mean(dim=3)  # (B, L, ngroups, d_state//2)
        sin_g = sin_g.mean(dim=3)

        # Apply RoPE to B and C
        B = _apply_rope(B, cos_g, sin_g)
        C = _apply_rope(C, cos_g, sin_g)

        # Flatten back
        B = B.view(batch, seqlen, self.ngroups * self.d_state)
        C = C.view(batch, seqlen, self.ngroups * self.d_state)

        return B, C

    def _apply_trapezoidal_blending(self, x, B, lam):
        """Apply trapezoidal discretization by blending current and previous inputs.
        
        Trapezoidal rule: effective input = λ_t * B_t * x_t + (1-λ_t) * B_{t-1} * x_{t-1}
        Implemented as pre-processing on x and B before the standard Euler scan.
        
        x: (batch, seqlen, d_ssm) or (batch, seqlen, nheads, headdim)
        B: (batch, seqlen, ngroups * d_state)
        lam: (batch, seqlen, nheads)  — trapezoidal gate λ_t = σ(u_t)
        
        When λ=1: standard Euler (Mamba-2 behavior)
        When λ=0.5: standard trapezoidal rule
        """
        # Blend x: x_eff[t] = λ_t * x[t] + (1-λ_t) * x[t-1]
        if x.dim() == 4:
            # (B, L, H, P) — expand lambda
            lam_x = lam.unsqueeze(-1)  # (B, L, H, 1)
        else:
            # (B, L, D) — expand lambda across d_ssm dimension
            lam_x = lam.repeat_interleave(self.headdim, dim=-1).unsqueeze(-1) if x.dim() == 3 else lam

        x_prev = F.pad(x[:, :-1], (0, 0, 1, 0) if x.dim() <= 3 else (0, 0, 0, 0, 1, 0))
        if x.dim() == 4:
            x_eff = lam_x * x + (1 - lam_x) * x_prev
        else:
            lam_expanded = lam.repeat_interleave(self.headdim, dim=-1)
            x_eff = lam_expanded * x + (1 - lam_expanded) * x_prev

        # Blend B: B_eff[t] = λ_t * B[t] + (1-λ_t) * B[t-1]
        B_prev = F.pad(B[:, :-1], (0, 0, 1, 0))
        # Expand lambda to match B shape
        # B is (B, L, ngroups * d_state), lam is (B, L, nheads)
        # Map nheads -> ngroups: average across heads in each group
        heads_per_group = self.nheads // self.ngroups
        lam_g = lam.view(lam.shape[0], lam.shape[1], self.ngroups, heads_per_group).mean(dim=-1)
        lam_B = lam_g.repeat_interleave(self.d_state, dim=-1)  # (B, L, ngroups * d_state)
        B_eff = lam_B * B + (1 - lam_B) * B_prev

        return x_eff, B_eff

    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
        """
        u: (batch, seqlen, d_model) if seqlen=None.
        Returns: same shape as u
        """
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        # ── Input projection (no conv1d in Mamba-3) ──
        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)

        # ── Split projections ──
        parts = self._split_projection(zxbcdt)
        idx = 0
        z = parts[idx]; idx += 1
        x = parts[idx]; idx += 1
        B = parts[idx]; idx += 1
        C = parts[idx]; idx += 1
        dt = parts[idx]; idx += 1

        lam = None
        if self.use_trapezoidal:
            lam_raw = parts[idx]; idx += 1
            lam = torch.sigmoid(lam_raw)  # λ_t = σ(u_t), (B, L, nheads)

        theta_raw = None
        if self.use_rope:
            theta_raw = parts[idx]; idx += 1

        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads,)

        # ── BC processing: bias + QK-norm ──
        B, C = self._apply_bc_processing(B, C)

        # ── Data-dependent RoPE (complex SSM) ──
        if self.use_rope and theta_raw is not None:
            B, C = self._apply_data_dependent_rope(B, C, theta_raw)

        # ── SiLU activation on x (replaces conv1d + activation) ──
        # Use F.silu (out-of-place) since x is a view from torch.split
        x = F.silu(x.clone())

        # ── MIMO: reshape x ──
        if self.mimo_rank > 1:
            # x: (B, L, d_ssm * r) -> process r copies through scan, then merge
            x_mimo = x.view(batch, seqlen, self.mimo_rank, self.d_ssm)
        else:
            x_mimo = x.unsqueeze(2)  # (B, L, 1, d_ssm) for uniform handling

        # ── Trapezoidal blending ──
        if self.use_trapezoidal and lam is not None:
            # Apply trapezoidal blending to each MIMO rank
            x_blended_list = []
            for r_idx in range(self.mimo_rank):
                x_r = x_mimo[:, :, r_idx, :]  # (B, L, d_ssm)
                if r_idx == 0:
                    x_r_eff, B_eff = self._apply_trapezoidal_blending(x_r, B, lam)
                else:
                    x_r_eff, _ = self._apply_trapezoidal_blending(x_r, B, lam)
                x_blended_list.append(x_r_eff)
            x_mimo = torch.stack(x_blended_list, dim=2)  # (B, L, r, d_ssm)
            B = B_eff  # Use blended B
        
        # ── Run SSM scan for each MIMO rank ──
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        
        # Ensure all tensors share the same dtype for Triton kernels (AMP may mix fp16/fp32)
        scan_dtype = x_mimo.dtype
        B = B.to(scan_dtype)
        C = C.to(scan_dtype)
        dt = dt.to(scan_dtype)

        y_list = []
        for r_idx in range(self.mimo_rank):
            x_r = x_mimo[:, :, r_idx, :]  # (B, L, d_ssm)

            y_r = mamba_chunk_scan_combined(
                rearrange(x_r, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                z=None,  # gate applied after norm in Mamba-3
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                **dt_limit_kwargs,
            )
            y_r = rearrange(y_r, "b l h p -> b l (h p)")
            y_list.append(y_r)

        # ── MIMO merge ──
        if self.mimo_rank > 1:
            y = torch.cat(y_list, dim=-1)  # (B, L, d_ssm * r)
            y = self.mimo_out_proj(y)       # (B, L, d_ssm)
        else:
            y = y_list[0]

        # ── Output norm + gate ──
        if self.rmsnorm:
            y = self.norm(y, z)
        else:
            y = y * self.act(z)

        if seqlen_og is not None:
            y = rearrange(y, "b l d -> (b l) d")
        out = self.out_proj(y)

        if self.process_group is not None:
            reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
            out = reduce_fn(out, self.process_group)

        return out

    def step(self, hidden_states, ssm_state):
        """Single-step decode (no conv state needed — Mamba-3 has no conv1d)."""
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time"
        
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B, d_in_proj)
        
        # Split — same as forward but no sequence dim
        parts = self._split_projection(zxbcdt)
        idx = 0
        z = parts[idx]; idx += 1
        x = parts[idx]; idx += 1
        B = parts[idx]; idx += 1
        C = parts[idx]; idx += 1
        dt = parts[idx]; idx += 1
        
        if self.use_trapezoidal:
            idx += 1  # skip lambda (not used in single-step decode)
        if self.use_rope:
            theta_raw = parts[idx]; idx += 1
        else:
            theta_raw = None

        # BC processing
        B, C = self._apply_bc_processing(B, C)

        A = -torch.exp(self.A_log.float())

        # SiLU activation on x
        x = self.act(x).to(dtype=dtype)

        # For single-step, MIMO processes each rank
        assert self.ngroups == 1, "Only support ngroups=1 for step decode"
        
        dt_val = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
        dA = torch.exp(dt_val * A)  # (batch, nheads)

        y_list = []
        for r_idx in range(self.mimo_rank):
            x_r = x[:, r_idx * self.d_ssm : (r_idx + 1) * self.d_ssm]
            x_r = rearrange(x_r, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt_val, B, x_r)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y_r = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y_r = y_r + rearrange(self.D.to(dtype), "h -> h 1") * x_r
            y_r = rearrange(y_r, "b h p -> b (h p)")
            y_list.append(y_r)

        if self.mimo_rank > 1:
            y = torch.cat(y_list, dim=-1)
            y = self.mimo_out_proj(y)
        else:
            y = y_list[0]

        if self.rmsnorm:
            y = self.norm(y, z)
        else:
            y = y * self.act(z)

        out = self.out_proj(y)
        return out.unsqueeze(1), ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        # No conv state needed in Mamba-3
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state,
            device=device, dtype=ssm_dtype,
        )
        return ssm_state
