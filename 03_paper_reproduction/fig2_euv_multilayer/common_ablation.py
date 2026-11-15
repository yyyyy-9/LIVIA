# -*- coding: utf-8 -*-
"""EUV multilayer parameterization, Sobol sampling, and batch evaluation."""

import sys, os, time, json, math
import numpy as np
import torch
from scipy.stats import qmc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '01_core_operators'))
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from python_torch_smm.tensor_smm_torch import fast_batch_smatrix_vd_torch as smm

LAMBDA_NM = 13.5
EDGE_DEG = 13.5
TMAX_DEG = 14.0

N_CL = 1.0
N_SI = 0.999002 + 0.001826j
N_MO = 0.923794 + 0.006435j
N_SUB = 0.978033 + 0.010773j
N_RU = 0.886360 + 0.017065j


def periodic_709_design(num_pairs=40, d_mo=3.32, d_si=3.77):
    """Periodic 7.09 nm Mo/Si multilayer thicknesses (80 layers)."""
    return np.array([d_mo, d_si] * num_pairs, dtype=np.float64)


def make_n_list(num_pairs=40):
    """Refractive index sequence: vacuum, Ru cap (0 nm placeholder), Mo/Si pairs, substrate."""
    return [N_CL, N_RU] + [N_MO, N_SI] * num_pairs + [N_SUB]


def make_theta_kx(tmax_deg=13.5, num_points=None):
    """Construct incidence angle grid (deg) and transverse wavevectors (nm^-1)."""
    if num_points is None:
        num_points = int(tmax_deg * 10) + 1
    theta = torch.linspace(0.0, tmax_deg, num_points, dtype=torch.float64)
    k0 = 2 * torch.pi / LAMBDA_NM
    kx = k0 * torch.sin(torch.deg2rad(theta))
    return theta, kx


def evaluate_design(thickness, num_pairs=40, tmax_deg=TMAX_DEG, edge_deg=EDGE_DEG, device=None):
    """Evaluate EUV Mo/Si reflector metrics (R0, dR, PDPS, C2/C4, SP)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d_layers = np.concatenate([[0.0], np.asarray(thickness, dtype=np.float64).flatten()])
    d_matrix = torch.tensor(d_layers, dtype=torch.float64, device=device).reshape(1, -1)
    n_list = make_n_list(num_pairs)
    theta_deg, kx_vec = make_theta_kx(tmax_deg)
    theta_deg = theta_deg.to(device)
    kx_vec = kx_vec.to(device)

    s11_te = smm(kx_vec, LAMBDA_NM, d_matrix, n_list, "TE")[0].flatten()
    s11_tm = smm(kx_vec, LAMBDA_NM, d_matrix, n_list, "TM")[0].flatten()
    s11_tm_aligned = -s11_tm

    r_te = torch.abs(s11_te) ** 2
    r_tm = torch.abs(s11_tm) ** 2
    r_avg = 0.5 * (r_te + r_tm)

    phi_te = -torch.rad2deg(torch.angle(s11_te * torch.conj(s11_te[:1])))
    phi_tm = -torch.rad2deg(torch.angle(s11_tm_aligned * torch.conj(s11_tm_aligned[:1])))

    idx_edge = int(torch.argmin(torch.abs(theta_deg - edge_deg)).item())
    basis_mat = torch.stack(
        [torch.ones_like(theta_deg), theta_deg, theta_deg ** 2, theta_deg ** 4], dim=1
    )
    coeffs_te = torch.linalg.lstsq(basis_mat, phi_te.unsqueeze(1).double()).solution
    coeffs_tm = torch.linalg.lstsq(basis_mat, phi_tm.unsqueeze(1).double()).solution

    r_avg_edge = r_avg[theta_deg <= edge_deg + 1e-12]
    return {
        "dR": float((r_avg_edge.max() - r_avg_edge.min()).item() * 100),
        "PDPS": float(torch.abs(phi_te[idx_edge] - phi_tm[idx_edge]).item()),
        "R0": float(r_te[0].item() * 100),
        "R0_TE": float(r_te[0].item() * 100),
        "R0_TM": float(r_tm[0].item() * 100),
        "C2_TE": float(coeffs_te[2].item()),
        "C2_TM": float(coeffs_tm[2].item()),
        "C4_TE": float(coeffs_te[3].item()),
        "C4_TM": float(coeffs_tm[3].item()),
        "SP": float(torch.abs(r_te[idx_edge] - r_tm[idx_edge]).item() * 100),
    }


def interpolate_control_points(ctrl_values, out_len):
    """Linearly interpolate low-dimensional control points onto per-bilayer profiles."""
    if ctrl_values.shape[1] == out_len:
        return ctrl_values
    return torch.nn.functional.interpolate(
        ctrl_values.unsqueeze(1), size=out_len, mode="linear", align_corners=True
    ).squeeze(1)


def compute_global_gamma_bounds(layer_min, layer_max, period_min, period_max, device):
    """Compute feasible Mo-fraction bounds given individual layer and period limits."""
    periods = torch.tensor([period_min, period_max], dtype=torch.float64, device=device)
    lower = torch.maximum(layer_min / periods, 1.0 - layer_max / periods)
    upper = torch.minimum(layer_max / periods, 1.0 - layer_min / periods)
    gamma_min = float(torch.max(lower).item())
    gamma_max = float(torch.min(upper).item())
    if gamma_min >= gamma_max:
        raise ValueError(f"No feasible gamma range: [{gamma_min:.4f}, {gamma_max:.4f}]")
    return gamma_min, gamma_max


def decode_graded_controls(
    W_ctrl, num_ctrl, period_min, period_max, gamma_min, gamma_max, num_pairs, period_monotonic=False
):
    """Map unconstrained control parameters (B, 2*num_ctrl) to Mo/Si thicknesses (B, 2*num_pairs)."""
    W_period = W_ctrl[:, :num_ctrl]
    W_gamma = W_ctrl[:, num_ctrl:]

    ctrl_periods = period_min + (period_max - period_min) * torch.sigmoid(W_period)
    if period_monotonic:
        ctrl_periods, _ = torch.sort(ctrl_periods, dim=1)

    ctrl_gamma_unit, _ = torch.sort(torch.sigmoid(W_gamma), dim=1)
    ctrl_gamma = gamma_min + (gamma_max - gamma_min) * ctrl_gamma_unit

    pair_periods = interpolate_control_points(ctrl_periods, num_pairs)
    pair_gamma = interpolate_control_points(ctrl_gamma, num_pairs)

    d_mosi = torch.zeros((W_ctrl.shape[0], 2 * num_pairs), dtype=torch.float64, device=W_ctrl.device)
    d_mosi[:, 0::2] = pair_periods * pair_gamma
    d_mosi[:, 1::2] = pair_periods * (1.0 - pair_gamma)
    return d_mosi, pair_periods, pair_gamma, ctrl_periods, ctrl_gamma

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NP, NC = 40, 16
NV_CTRL = 2 * NC
NV_80D  = 80
NV_32D  = 32

NL = make_n_list(NP)
TH, KX = make_theta_kx(13.5)
TH = TH.to(DEV)
KX = KX.to(DEV)
EI = int(torch.argmin(torch.abs(TH - EDGE_DEG)).item())

BS = 4096


# ═══════════════════════════════════════════════════════════════
# Phase parameters
# ═══════════════════════════════════════════════════════════════

PHASE_PARAMS = {
    'narrow': {
        'LMIN': 2.50, 'LMAX': 4.84,
        'PMI': 5.80, 'PMA': 8.50,
        'gm':  0.431, 'gM': 0.569,
        'PR':  2.70, 'GR': 0.138,
    },
    'wide': {
        'LMIN': 2.50, 'LMAX': 7.92,
        'PMI': 5.30, 'PMA': 15.00,
        'gm':  0.472, 'gM': 0.528,
        'PR':  9.70, 'GR': 0.056,
    },
}

# Periodic bottom for 32D_top experiments (period 3.32/3.77, 24 pairs)
PERIODIC_BOTTOM = np.array([3.32, 3.77] * 24, dtype=np.float64)  # 48 elements


# ═══════════════════════════════════════════════════════════════
# Sobol generation
# ═══════════════════════════════════════════════════════════════

def generate_sobol_80D(seed, n=16777216):
    """Generate (n, 80) Sobol points in [0,1]^80."""
    sampler = qmc.Sobol(d=NV_80D, scramble=True, seed=seed)
    m = int(math.ceil(math.log2(max(n, 2))))
    return sampler.random_base2(m)[:n].astype(np.float64)

def generate_sobol_32D(seed, n=16777216):
    """Generate (n, 32) Sobol points in [0,1]^32."""
    sampler = qmc.Sobol(d=NV_32D, scramble=True, seed=seed)
    m = int(math.ceil(math.log2(max(n, 2))))
    return sampler.random_base2(m)[:n].astype(np.float64)


# ═══════════════════════════════════════════════════════════════
# Layer mapping functions (Sobol → 80-layer d_mosi)
# ═══════════════════════════════════════════════════════════════

def map_80D_independent(u, phase):
    """u: (B, 80) in [0,1]. Returns d_mosi: (B, 80)."""
    LMIN = PHASE_PARAMS[phase]['LMIN']
    LMAX = PHASE_PARAMS[phase]['LMAX']
    return LMIN + u * (LMAX - LMIN)

def map_32D_ctrl(u, phase):
    """u: (B, 32) in [0,1]. Returns d_mosi: (B, 80) via graded control point decode."""
    PMI = PHASE_PARAMS[phase]['PMI']
    PMA = PHASE_PARAMS[phase]['PMA']
    gm  = PHASE_PARAMS[phase]['gm']
    gM  = PHASE_PARAMS[phase]['gM']
    PR  = PHASE_PARAMS[phase]['PR']
    GR  = PHASE_PARAMS[phase]['GR']

    B = u.shape[0]
    # Map to period and gamma
    periods_np = PMI + u[:, :NC] * PR
    gammas_np  = gm  + u[:, NC:] * GR
    gammas_np  = np.sort(gammas_np, axis=1)  # monotonic gamma

    ctrl_periods = torch.tensor(periods_np, dtype=torch.float64, device=DEV)
    ctrl_gamma   = torch.tensor(gammas_np, dtype=torch.float64, device=DEV)

    pair_periods = interpolate_control_points(ctrl_periods, NP)
    pair_gamma   = interpolate_control_points(ctrl_gamma, NP)

    d_mosi = torch.zeros((B, 2 * NP), dtype=torch.float64, device=DEV)
    d_mosi[:, 0::2] = pair_periods * pair_gamma
    d_mosi[:, 1::2] = pair_periods * (1.0 - pair_gamma)
    return d_mosi.cpu().numpy()

def map_32D_top(u, phase):
    """u: (B, 32) in [0,1]. Top 32 layers random, bottom 48 = periodic [3.32,3.77]^24."""
    LMIN = PHASE_PARAMS[phase]['LMIN']
    LMAX = PHASE_PARAMS[phase]['LMAX']

    B = u.shape[0]
    top = LMIN + u[:, :32] * (LMAX - LMIN)  # (B, 32)
    bottom = np.tile(PERIODIC_BOTTOM, (B, 1))  # (B, 48)
    return np.concatenate([top, bottom], axis=1)  # (B, 80)


# ═══════════════════════════════════════════════════════════════
# SMM evaluation (batch, returns R0, dR, PDPS)
# ═══════════════════════════════════════════════════════════════

def _smm_safe(kx, wl, df, nl, pol):
    """SMM wrapper with automatic batch-size fallback on OOM."""
    try:
        return smm(kx, wl, df, nl, pol)
    except torch.cuda.OutOfMemoryError:
        B = df.shape[0]
        if B <= 256:
            raise
        torch.cuda.empty_cache()
        mid = B // 2
        s1, s2, s3, s4 = _smm_safe(kx, wl, df[:mid], nl, pol)
        t1, t2, t3, t4 = _smm_safe(kx, wl, df[mid:], nl, pol)
        return (torch.cat([s1, t1], dim=0), torch.cat([s2, t2], dim=0),
                torch.cat([s3, t3], dim=0), torch.cat([s4, t4], dim=0))

@torch.no_grad()
def evaluate_batch(d_mosi_np):
    """Evaluate a batch of designs: returns R0(%), dR(%), PDPS(deg)."""
    B = d_mosi_np.shape[0]
    d_mosi = torch.tensor(d_mosi_np, dtype=torch.float64, device=DEV)
    df = torch.cat([torch.zeros(B, 1, dtype=torch.float64, device=DEV), d_mosi], dim=1)

    s11e = _smm_safe(KX, LAMBDA_NM, df, NL, 'TE')[0]
    s11m = _smm_safe(KX, LAMBDA_NM, df, NL, 'TM')[0]
    ra = 0.5 * (torch.abs(s11e) ** 2 + torch.abs(s11m) ** 2)

    R0_b   = ra[:, 0].cpu().numpy() * 100.0
    dR_b   = (ra.max(dim=1).values.cpu().numpy() - ra.min(dim=1).values.cpu().numpy()) * 100.0

    ph_te = -torch.rad2deg(torch.angle(s11e * torch.conj(s11e[:, :1])))
    ph_tm = -torch.rad2deg(torch.angle(s11m * torch.conj(s11m[:, :1])))
    pdps_b = torch.abs(ph_te[:, EI] - ph_tm[:, EI]).cpu().numpy()

    return R0_b, dR_b, pdps_b


# ═══════════════════════════════════════════════════════════════
# Periodic baseline check
# ═══════════════════════════════════════════════════════════════

def check_periodic_baseline():
    """Verify that the periodic Mo/Si baseline reproduces dR = 8.71%."""
    p709 = periodic_709_design()
    r = evaluate_design(p709, device=DEV)
    R0_b, dR_b, pdps_b = evaluate_batch(p709.reshape(1, -1))
    ok = abs(r['dR'] - 8.71) < 0.5 and abs(dR_b[0] - 8.71) < 0.5
    print(f"Periodic baseline: evaluate_design dR={r['dR']:.2f}%, evaluate_batch dR={dR_b[0]:.2f}%")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ═══════════════════════════════════════════════════════════════
# Logging helper
# ═══════════════════════════════════════════════════════════════

def log_event(log_path, msg):
    """Append timestamped message to log file and print."""
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
