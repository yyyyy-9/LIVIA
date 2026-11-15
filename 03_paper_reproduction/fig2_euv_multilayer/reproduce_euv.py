#!/usr/bin/env python3
"""
reproduce_euv.py — EUV module reproduction.
Reproduces the EUV forward-sampling statistics, AD refinement outcomes,
and AD/FD gradient-accuracy verification reported in §3.1.1 and SI §5.

Usage:
    python reproduce_euv.py --phase forward   # Forward scan only (~7h)
    python reproduce_euv.py --phase aggregate # Paper statistics (4 primary seeds)
    python reproduce_euv.py --phase refine    # AD refinement only (~2h)
    python reproduce_euv.py --phase adfd      # AD/FD verification only
    python reproduce_euv.py --phase all       # Everything with paper seeds (~8h)
    python reproduce_euv.py --phase verify    # Verify existing outputs
    python reproduce_euv.py --phase verify --generated # Verify a local rerun
    python reproduce_euv.py --phase aggregate --seed-set all # Aggregate all 6 seeds
"""
import sys, os, time, math, json, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common_ablation import (
    DEV, NP, NC, NV_80D, NV_32D, BS,
    NL, KX, LAMBDA_NM, TH, EI,
    generate_sobol_80D, generate_sobol_32D,
    map_80D_independent, map_32D_ctrl, map_32D_top,
    evaluate_batch, _smm_safe, PHASE_PARAMS,
    interpolate_control_points,
)

OUT_DIR = HERE
DATA_DIR = os.path.join(OUT_DIR, 'data')
REFERENCE_DIR = os.path.join(OUT_DIR, 'outputs')
OUTPUTS_DIR = os.path.join(REFERENCE_DIR, 'generated')

N_PER_SEED = 16777216
PRIMARY_SEEDS = [42, 123, 456, 789]
VALIDATION_SEEDS = [7, 2024]
ALL_SEEDS = PRIMARY_SEEDS + VALIDATION_SEEDS
ARCHS = ['80D', '32D_ctrl', '32D_top']
PHASES = ['narrow', 'wide']
CHECKPOINTS = [524288, 1048576, 2097152, 4194304, 8388608, 16777216]
EDGES_R0 = np.linspace(31, 70, 79)
EDGES_DR = np.linspace(0, 50, 51)
EDGES_PDPS = np.linspace(0, 20, 41)

def log_event(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════
# Phase 1: Forward scan
# ═══════════════════════════════════════════════════════

def get_config(arch):
    if arch == '80D':
        return {'nv': NV_80D, 'gen': generate_sobol_80D, 'map': map_80D_independent}
    elif arch == '32D_ctrl':
        return {'nv': NV_32D, 'gen': generate_sobol_32D, 'map': map_32D_ctrl}
    else:
        return {'nv': NV_32D, 'gen': generate_sobol_32D, 'map': map_32D_top}

def run_forward(arch, phase, seed, skip_if_exists=False):
    cfg = get_config(arch)
    gen_fn, map_fn = cfg['gen'], cfg['map']
    sd = os.path.join(DATA_DIR, f'phase_{phase}', arch)
    os.makedirs(sd, exist_ok=True)
    full_path = os.path.join(sd, f'full_seed{seed}_16M.npz')

    if skip_if_exists and os.path.isfile(full_path):
        log_event(f"SKIP: {arch}_{phase}_seed{seed} (exists)")
        d = np.load(full_path, allow_pickle=True)
        return {'arch': arch, 'phase': phase, 'seed': seed,
                'p64': float(np.mean(d['R0'] >= 64.0) * 100),
                'elite': int(np.sum(d['R0'] >= 64.0))}

    t0 = time.time()
    log_event(f"START: {arch}_{phase}_seed{seed}")
    u_all = gen_fn(seed, N_PER_SEED)
    idx_arr = np.zeros(N_PER_SEED, dtype=np.int32)
    R0_arr = np.zeros(N_PER_SEED, dtype=np.float32)
    dR_arr = np.zeros(N_PER_SEED, dtype=np.float32)
    PDPS_arr = np.zeros(N_PER_SEED, dtype=np.float32)
    elite_idx, elite_thick, elite_R0v, elitedR_v, elitePDPS_v = [], [], [], [], []
    H_rd = np.zeros((len(EDGES_R0)-1, len(EDGES_DR)-1), dtype=np.int64)
    H_rp = np.zeros((len(EDGES_R0)-1, len(EDGES_PDPS)-1), dtype=np.int64)
    N_done = 0

    for gc in range(math.ceil(N_PER_SEED / BS)):
        s, e = gc * BS, min((gc + 1) * BS, N_PER_SEED)
        thickness = map_fn(u_all[s:e], phase)
        R0_b, dR_b, PDPS_b = evaluate_batch(thickness)
        idx_arr[s:e] = np.arange(s, e, dtype=np.int32)
        R0_arr[s:e] = R0_b.astype(np.float32)
        dR_arr[s:e] = dR_b.astype(np.float32)
        PDPS_arr[s:e] = PDPS_b.astype(np.float32)
        em = R0_b >= 64.0
        if em.any():
            elite_idx.append(np.arange(s, e, dtype=np.int32)[em])
            elite_thick.append(thickness[em].astype(np.float32))
            elite_R0v.append(R0_b[em].astype(np.float32))
            elitedR_v.append(dR_b[em].astype(np.float32))
            elitePDPS_v.append(PDPS_b[em].astype(np.float32))
        H_rd += np.histogram2d(R0_b, dR_b, bins=[EDGES_R0, EDGES_DR])[0].astype(np.int64)
        H_rp += np.histogram2d(R0_b, PDPS_b, bins=[EDGES_R0, EDGES_PDPS])[0].astype(np.int64)
        N_done += e - s
        for cp in CHECKPOINTS:
            if N_done == cp:
                log_event(f"  {arch}_{phase}_seed{seed} cp {cp//1024}K: {(N_done/(time.time()-t0)):.0f} pts/s")

    elapsed = time.time() - t0
    np.savez_compressed(full_path, idx=idx_arr, R0=R0_arr, dR=dR_arr, PDPS=PDPS_arr)
    n_elite = 0
    if elite_idx:
        np.savez_compressed(os.path.join(sd, f'elite_seed{seed}_16M.npz'),
            idx=np.concatenate(elite_idx), thickness=np.concatenate(elite_thick, axis=0),
            R0=np.concatenate(elite_R0v), dR=np.concatenate(elitedR_v),
            PDPS=np.concatenate(elitePDPS_v))
        n_elite = len(np.concatenate(elite_idx))
    np.savez_compressed(os.path.join(sd, f'hist_seed{seed}_16M.npz'),
        H_rd=H_rd, H_rp=H_rp, N=N_PER_SEED,
        edges_r0=EDGES_R0, edges_dr=EDGES_DR, edges_pdps=EDGES_PDPS)

    p64 = float(np.mean(R0_arr >= 64.0) * 100)
    p68 = float(np.mean(R0_arr >= 68.0) * 100)
    log_event(f"DONE: {arch}_{phase}_seed{seed}: {elapsed:.0f}s elite={n_elite} P64={p64:.4f}% P68={p68:.4f}%")
    torch.cuda.empty_cache()
    return {'arch': arch, 'phase': phase, 'seed': seed, 'elapsed': elapsed,
            'p64': p64, 'p68': p68, 'elite': n_elite}


# ═══════════════════════════════════════════════════════
# Phase 2: Aggregate
# ═══════════════════════════════════════════════════════

def run_aggregate(seeds=None):
    if seeds is None:
        seeds = PRIMARY_SEEDS
    survey_paths = [
        os.path.join(DATA_DIR, f'phase_{phase}', arch, f'full_seed{seed}_16M.npz')
        for phase in PHASES for arch in ARCHS for seed in seeds
    ]
    missing = [path for path in survey_paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            f'Missing {len(missing)} survey files. Run --phase forward first. First missing: {missing[0]}'
        )
    pooled = {}
    pdps_cdfs = {}
    for arch in ARCHS:
        merged_R0, merged_dR, merged_pdps = [], [], []
        sd = os.path.join(DATA_DIR, 'phase_narrow', arch)
        for seed in seeds:
            fp = os.path.join(sd, f'full_seed{seed}_16M.npz')
            with np.load(fp) as dat:
                merged_R0.append(dat['R0'])
                merged_dR.append(dat['dR'])
                merged_pdps.append(dat['PDPS'])
        R0 = np.concatenate(merged_R0); dR = np.concatenate(merged_dR); pdps = np.concatenate(merged_pdps)
        mask = R0 >= 64.0
        if arch in ('32D_ctrl', '32D_top'):
            if not np.any(mask):
                raise ValueError(f'The conditional PDPS distribution is empty for {arch}')
            counts, edges = np.histogram(pdps[mask], bins=EDGES_PDPS)
            centers = (edges[:-1] + edges[1:]) / 2
            pdps_cdfs[arch] = np.column_stack((centers, np.cumsum(counts) / mask.sum() * 100))
        pooled[arch] = {
            'p64': float(np.mean(R0 >= 64.0) * 100),
            'p68': float(np.mean(R0 >= 68.0) * 100),
            'p_dr_lt10': float(np.mean(dR[mask] < 10.0) * 100),
            'p_pdps_lt7': float(np.mean(pdps[mask] < 7.0) * 100),
            'stack': {
                'both': float(np.mean((dR[mask] < 10.0) & (pdps[mask] < 7.0)) * 100),
                'dr_only': float(np.mean((dR[mask] < 10.0) & (pdps[mask] >= 7.0)) * 100),
                'pdps_only': float(np.mean((dR[mask] >= 10.0) & (pdps[mask] < 7.0)) * 100),
                'neither': float(np.mean((dR[mask] >= 10.0) & (pdps[mask] >= 7.0)) * 100)
            }
        }
    wide_counts = {}
    for arch in ARCHS:
        cnt = 0
        sd = os.path.join(DATA_DIR, 'phase_wide', arch)
        for seed in seeds:
            fp = os.path.join(sd, f'full_seed{seed}_16M.npz')
            with np.load(fp) as dat:
                cnt += int(np.sum(dat['R0'] >= 64.0))
        wide_counts[arch] = cnt

    data_json = {f'x{arch}': pooled[arch] for arch in ARCHS}
    data_json['wide'] = {f'x{arch}': wide_counts[arch] for arch in ARCHS}
    suffix = '_all' if seeds == ALL_SEEDS else ''
    with open(os.path.join(OUTPUTS_DIR, f'data{suffix}.json'), 'w') as f:
        json.dump(data_json, f, indent=2)
    for arch, cdf in pdps_cdfs.items():
        np.savetxt(os.path.join(OUTPUTS_DIR, f'pdps_cdf_{arch}{suffix}.csv'), cdf,
                   delimiter=',', header='pdps_center,cdf_pct', comments='')
    log_event(f"data{suffix}.json written: {list(pooled.keys())} seeds={len(seeds)}")


# ═══════════════════════════════════════════════════════
# Phase 3: AD Refinement (R0-weighted sorting)
# ═══════════════════════════════════════════════════════

def run_refine(seed):
    N_SOBOL = 2097152
    MH, MW = 0.60, 10.0
    R0_TARGET, PDPS_CEIL = 0.64, 7.0

    log_event(f"REFINE seed={seed}: 2M forward + 300 candidates + 450 iter")
    u = generate_sobol_32D(seed, N_SOBOL)
    good = []
    t0 = time.time()
    for ci in range(math.ceil(N_SOBOL / BS)):
        s, e = ci * BS, min((ci + 1) * BS, N_SOBOL)
        thickness = map_32D_ctrl(u[s:e], 'narrow')
        R0_b, dR_b, PDPS_b = evaluate_batch(thickness)
        for i in range(e - s):
            if dR_b[i] < 20.0 and R0_b[i] > 50.0:
                good.append({'dR': float(dR_b[i]), 'R0': float(R0_b[i]), 't': thickness[i].tolist()})

    cands = sorted(good, key=lambda x: max(0, 64.0 - x['R0']) * 3 + x['dR'])[:300]
    B = len(cands)
    log_event(f"  Selected {B} candidates, best dR={cands[0]['dR']:.2f}%")

    W_list = []
    for c in cands:
        tp = np.array(c['t'], dtype=np.float64)
        pp = tp[::2] + tp[1::2]; pg = tp[::2] / np.clip(pp, 1e-12, None)
        pc = interpolate_control_points(torch.tensor(pp, dtype=torch.float64).unsqueeze(0), NC)
        gc = interpolate_control_points(torch.tensor(pg, dtype=torch.float64).unsqueeze(0), NC)
        gc, _ = torch.sort(gc, dim=1)
        pu = (pc - 5.80) / 2.70; gu = (gc - 0.431) / 0.138
        W_list.append(torch.cat([
            torch.log(torch.clamp(pu, 1e-6, 1-1e-6) / (1 - torch.clamp(pu, 1e-6, 1-1e-6))),
            torch.log(torch.clamp(gu, 1e-6, 1-1e-6) / (1 - torch.clamp(gu, 1e-6, 1-1e-6)))], dim=1))
    W = torch.cat(W_list, dim=0).to(DEV).requires_grad_(True)
    opt = torch.optim.Adam([W], lr=0.002)

    for it in range(450):
        opt.zero_grad(set_to_none=True)
        pu = torch.sigmoid(W[:, :NC]); gu = torch.sigmoid(W[:, NC:]); gu, _ = torch.sort(gu, dim=1)
        pp = torch.nn.functional.interpolate((5.80 + pu * 2.70).unsqueeze(1), size=NP, mode='linear', align_corners=True).squeeze(1)
        pg = torch.nn.functional.interpolate((0.431 + gu * 0.138).unsqueeze(1), size=NP, mode='linear', align_corners=True).squeeze(1)
        d_mosi = torch.zeros((B, 2*NP), dtype=torch.float64, device=DEV)
        d_mosi[:, 0::2] = pp * pg; d_mosi[:, 1::2] = pp * (1.0 - pg)
        df = torch.cat([torch.zeros(B, 1, dtype=torch.float64, device=DEV), d_mosi], dim=1)
        s11e = _smm_safe(KX, LAMBDA_NM, df, NL, 'TE')[0]; s11m = _smm_safe(KX, LAMBDA_NM, df, NL, 'TM')[0]
        ra = 0.5 * (torch.abs(s11e)**2 + torch.abs(s11m)**2)
        dr = ra.max(dim=1).values - ra.min(dim=1).values
        progress = it / 450
        r0_w = 12.0 + 8.0 * progress
        loss_terms = dr + r0_w * torch.relu(R0_TARGET - ra[:, 0]) + MW * torch.relu(MH - ra.min(dim=1).values)
        ph_te = -torch.rad2deg(torch.angle(s11e * torch.conj(s11e[:, :1])))
        ph_tm = -torch.rad2deg(torch.angle(s11m * torch.conj(s11m[:, :1])))
        pdps_v = torch.abs(ph_te[:, EI] - ph_tm[:, EI])
        pdps_w = 2.0 * progress
        loss_terms = loss_terms + pdps_w * torch.nn.functional.softplus(pdps_v - PDPS_CEIL)
        loss = torch.mean(loss_terms); loss.backward(); opt.step()
        if it % 150 == 0:
            log_event(f"    iter {it}: feas={(ra[:,0]>=R0_TARGET).sum().item()}/{B}")

    with torch.no_grad():
        pu = torch.sigmoid(W[:, :NC]); gu = torch.sigmoid(W[:, NC:]); gu, _ = torch.sort(gu, dim=1)
        pp = torch.nn.functional.interpolate((5.80 + pu * 2.70).unsqueeze(1), size=NP, mode='linear', align_corners=True).squeeze(1)
        pg = torch.nn.functional.interpolate((0.431 + gu * 0.138).unsqueeze(1), size=NP, mode='linear', align_corners=True).squeeze(1)
        d_mosi = torch.zeros((B, 2*NP), dtype=torch.float64, device=DEV)
        d_mosi[:, 0::2] = pp * pg; d_mosi[:, 1::2] = pp * (1.0 - pg)
        df = torch.cat([torch.zeros(B, 1, dtype=torch.float64, device=DEV), d_mosi], dim=1)
        s11e = _smm_safe(KX, LAMBDA_NM, df, NL, 'TE')[0]; s11m = _smm_safe(KX, LAMBDA_NM, df, NL, 'TM')[0]
        ra = 0.5 * (torch.abs(s11e)**2 + torch.abs(s11m)**2)
        dr = ra.max(dim=1).values - ra.min(dim=1).values
        ph_te = -torch.rad2deg(torch.angle(s11e * torch.conj(s11e[:, :1])))
        ph_tm = -torch.rad2deg(torch.angle(s11m * torch.conj(s11m[:, :1])))
        pdps_v = torch.abs(ph_te[:, EI] - ph_tm[:, EI])
        R0_f = ra[:, 0].cpu().numpy(); dr_f = dr.cpu().numpy() * 100; pdps_f = pdps_v.cpu().numpy()
        f_mask = (R0_f >= 0.64) & (pdps_f <= 7.0)
        feasible = int(f_mask.sum())
        best_dR, best_PDPS, best_R0, best_t = 999, 999, 0, None
        if feasible > 0:
            fi = np.where(f_mask)[0]; bj = fi[np.argmin(dr_f[fi])]
            best_dR = float(dr_f[bj]); best_PDPS = float(pdps_f[bj]); best_R0 = float(R0_f[bj]) * 100
            best_t = d_mosi[bj].cpu().numpy().tolist()

    elapsed = time.time() - t0
    log_event(f"  DONE seed={seed}: dR={best_dR:.2f}% PDPS={best_PDPS:.2f}deg R0={best_R0:.2f}% feas={feasible} ({elapsed:.0f}s)")

    result = {'best_dR': best_dR, 'best_PDPS': best_PDPS, 'best_R0': best_R0,
              'feasible': feasible, 'thickness': best_t, 'seed': seed}
    with open(os.path.join(OUTPUTS_DIR, f'optimize_seed{seed}.json'), 'w') as f:
        json.dump(result, f, indent=2)
    return result


# ═══════════════════════════════════════════════════════
# Phase 4: AD/FD Verification
# ═══════════════════════════════════════════════════════

def run_adfd():
    log_event("AD/FD verification (logit-space perturbations, matching SI)")

    # Use periodic baseline, 80 Mo/Si layers (40 pairs)
    t_np = np.array([3.32, 3.77] * 40, dtype=np.float64)
    LMIN, LMAX = 2.50, 4.84

    # Convert to logit variables
    tu = np.clip((t_np - LMIN) / (LMAX - LMIN), 1e-6, 1 - 1e-6)
    w_np = np.log(tu / (1 - tu))
    W = torch.tensor(w_np, dtype=torch.float64, device=DEV).unsqueeze(0)
    W.requires_grad_(True)

    # AD gradient on logit variable
    dm = LMIN + (LMAX - LMIN) * torch.sigmoid(W)
    df = torch.cat([torch.zeros(1, 1, dtype=torch.float64, device=DEV), dm], dim=1)
    s11e = _smm_safe(KX, LAMBDA_NM, df, NL, 'TE')[0]
    s11m = _smm_safe(KX, LAMBDA_NM, df, NL, 'TM')[0]
    ra = 0.5 * (torch.abs(s11e)**2 + torch.abs(s11m)**2)
    lo = ra.max() - ra.min()
    lo.backward()
    grad_ad = W.grad[0].cpu().numpy().copy()

    results = []
    for h in [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 1e-5, 1e-6, 1e-7]:
        grad_fd = np.zeros(80)
        for i in range(80):
            for sgn in [1, -1]:
                Wx = W.detach().clone()
                Wx[0, i] += sgn * h
                with torch.no_grad():
                    dm_i = LMIN + (LMAX - LMIN) * torch.sigmoid(Wx)
                    df_i = torch.cat([torch.zeros(1, 1, dtype=torch.float64, device=DEV), dm_i], dim=1)
                    s11ei = _smm_safe(KX, LAMBDA_NM, df_i, NL, 'TE')[0]
                    s11mi = _smm_safe(KX, LAMBDA_NM, df_i, NL, 'TM')[0]
                    ra_i = 0.5 * (torch.abs(s11ei)**2 + torch.abs(s11mi)**2)
                    grad_fd[i] += sgn * (ra_i.max() - ra_i.min()).item()
            grad_fd[i] /= (2 * h)
        cs = np.dot(grad_ad, grad_fd) / (np.linalg.norm(grad_ad) * np.linalg.norm(grad_fd) + 1e-16)
        re = float(np.linalg.norm(grad_ad - grad_fd) / np.linalg.norm(grad_ad))
        results.append({'h': h, 'cos_sim': float(cs), 'rel_error': re})
        log_event(f"  AD/FD h={h:.0e}: cos_sim={cs:.16f} rel_error={re:.2e}")

    with open(os.path.join(OUTPUTS_DIR, 'ad_fd_result.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # AD/FD speedup scaling (D=5..80, matching SI Fig. S7b)
    log_event("AD/FD speedup scaling (D=5..80)")
    dims = [5, 10, 20, 40, 80]
    AD_SAMPLES, FD_SAMPLES = 20, 3
    scaling = []
    for N_idx, N in enumerate(dims):
        t_np_N = np.array([3.32, 3.77] * 40, dtype=np.float64)
        tu_N = np.clip((t_np_N - LMIN) / (LMAX - LMIN), 1e-6, 1 - 1e-6)

        # AD timing: one forward + backward pass
        ad_sum = 0.0
        for _ in range(AD_SAMPLES):
            WR = torch.randn(1, N, dtype=torch.float64, device=DEV, requires_grad=True)
            if DEV.type == 'cuda': torch.cuda.synchronize()
            t0_a = time.time()
            dmi = LMIN + (LMAX - LMIN) * torch.sigmoid(WR)
            dpad = torch.zeros(1, 80, dtype=torch.float64, device=DEV)
            dpad[0, :N] = dmi[0]
            dfi = torch.cat([torch.zeros(1, 1, dtype=torch.float64, device=DEV), dpad], dim=1)
            s1e = _smm_safe(KX, LAMBDA_NM, dfi, NL, 'TE')[0]
            s1m = _smm_safe(KX, LAMBDA_NM, dfi, NL, 'TM')[0]
            (0.5 * (torch.abs(s1e)**2 + torch.abs(s1m)**2)).max().backward()
            if DEV.type == 'cuda': torch.cuda.synchronize()
            ad_sum += time.time() - t0_a
        ad_time = ad_sum / AD_SAMPLES

        # FD timing: 2*N forward evaluations (central difference, step=1e-4)
        fd_sum = 0.0
        for _ in range(FD_SAMPLES):
            WR = torch.randn(1, N, dtype=torch.float64, device=DEV)
            if DEV.type == 'cuda': torch.cuda.synchronize()
            t0_f = time.time()
            for i in range(N):
                for sgn in [1, -1]:
                    Wx = WR.clone()
                    Wx[0, i] += sgn * 1e-4
                    with torch.no_grad():
                        dmi2 = LMIN + (LMAX - LMIN) * torch.sigmoid(Wx)
                        dpad2 = torch.zeros(1, 80, dtype=torch.float64, device=DEV)
                        dpad2[0, :N] = dmi2[0]
                        dfi2 = torch.cat([torch.zeros(1, 1, dtype=torch.float64, device=DEV), dpad2], dim=1)
                        s1ei = _smm_safe(KX, LAMBDA_NM, dfi2, NL, 'TE')[0]
                        s1mi = _smm_safe(KX, LAMBDA_NM, dfi2, NL, 'TM')[0]
                        _ = (0.5 * (torch.abs(s1ei)**2 + torch.abs(s1mi)**2)).max()
            if DEV.type == 'cuda': torch.cuda.synchronize()
            fd_sum += time.time() - t0_f
        fd_time = fd_sum / FD_SAMPLES

        speedup = fd_time / ad_time
        scaling.append({'N': N, 'AD_time_s': round(ad_time, 4), 'FD_time_s': round(fd_time, 4),
                        'speedup': round(speedup, 2)})
        log_event(f"  N={N:3d}: AD={ad_time:.4f}s FD={fd_time:.2f}s speedup={speedup:.1f}x")

    with open(os.path.join(OUTPUTS_DIR, 'ad_fd_scaling.json'), 'w') as f:
        json.dump(scaling, f, indent=2)
    log_event(f"  Saved: ad_fd_scaling.json")


# ═══════════════════════════════════════════════════════
# Phase 5: Verification Report
# ═══════════════════════════════════════════════════════

def run_verify(input_dir, seed_set):
    """Print verification report comparing outputs against paper values."""
    log_event("=== VERIFICATION REPORT ===")
    ok, bad = 0, 0
    summary_name = 'data_all.json' if seed_set == 'all' else 'data.json'
    required = [summary_name, 'ad_fd_result.json']
    required += [f'optimize_seed{seed}.json' for seed in ALL_SEEDS]
    missing = [name for name in required if not os.path.isfile(os.path.join(input_dir, name))]
    for name in missing:
        log_event(f'  Missing {name} [FAIL]')
        bad += 1

    # P64
    fp = os.path.join(input_dir, summary_name)
    if os.path.isfile(fp):
        dj = json.load(open(fp))
        paper_p64 = {'x80D': 0.113, 'x32D_ctrl': 1.02, 'x32D_top': 2.17}
        for k, pv in paper_p64.items():
            mv = dj[k]['p64']
            d = abs(mv - pv)
            s = 'OK' if d < 0.01 else 'FAIL'
            line = f"  P64 {k}: paper={pv}% measured={mv:.4f}% delta={d:.4f} [{s}]"
            log_event(line)
            if s == 'OK': ok += 1
            else: bad += 1

        # PDPS<7
        paper_pdps = {'x32D_ctrl': 59.4, 'x32D_top': 9.2}
        for k, pv in paper_pdps.items():
            mv = dj[k]['p_pdps_lt7']
            d = abs(mv - pv)
            s = 'OK' if d < 0.5 else 'FAIL'
            line = f"  PDPS<7 {k}: paper={pv}% measured={mv:.2f}% delta={d:.2f} [{s}]"
            log_event(line)
            if s == 'OK': ok += 1
            else: bad += 1

        # Wide
        w = dj['wide']
        log_event(f"  Wide elite (primary): 80D={w['x80D']} ctrl={w['x32D_ctrl']} top={w['x32D_top']}")

        # If validation data exists, show it too
        fp2 = os.path.join(input_dir, 'data_all.json')
        if seed_set == 'paper' and os.path.isfile(fp2):
            dj2 = json.load(open(fp2))
            w2 = dj2['wide']
            log_event(f"  Wide elite (all 6 seeds): 80D={w2['x80D']} ctrl={w2['x32D_ctrl']} top={w2['x32D_top']}")

    # AD refinement
    ad_drs, ad_pdps, ad_r0s = [], [], []
    for s in ALL_SEEDS:
        fp = os.path.join(input_dir, f'optimize_seed{s}.json')
        if os.path.isfile(fp):
            oj = json.load(open(fp))
            ad_drs.append(oj['best_dR']); ad_pdps.append(oj['best_PDPS']); ad_r0s.append(oj['best_R0'])
            full_ok = (oj['best_R0'] >= 64.0 and oj['best_PDPS'] <= 5.0 and
                       2.5 <= oj['best_dR'] <= 3.0 and oj['feasible'] > 0)
            ok_s = 'OK' if full_ok else 'FAIL'
            line = f"  refine seed{s}: dR={oj['best_dR']:.2f}% PDPS={oj['best_PDPS']:.2f}deg R0={oj['best_R0']:.2f}% feas={oj['feasible']} [{ok_s}]"
            log_event(line)
            if ok_s == 'OK': ok += 1
            else: bad += 1

    if ad_drs:
        def cv(arr):
            return float(np.std(arr) / np.mean(arr) * 100)
        log_event(f"  AD cross-seed CV: dR={cv(ad_drs):.1f}% PDPS={cv(ad_pdps):.1f}% R0={cv(ad_r0s):.2f}%")

    # AD/FD
    fp = os.path.join(input_dir, 'ad_fd_result.json')
    if os.path.isfile(fp):
        af = json.load(open(fp))
        steps = {0.01, 0.005, 0.001, 0.0005, 0.0001, 1e-5, 1e-6, 1e-7}
        if len(af) != len(steps) or {entry['h'] for entry in af} != steps:
            log_event('  Incomplete AD/FD step sweep [FAIL]')
            bad += 1
        for e in af:
            s = 'OK' if e['cos_sim'] > 0.999999 else 'FAIL'
            line = f"  AD/FD h={e['h']:.0e}: cos={e['cos_sim']:.16f} [{s}]"
            log_event(line)
            if s == 'OK': ok += 1
            else: bad += 1

    total_line = f"\n  TOTAL: {ok} OK / {ok+bad} checks"
    log_event(total_line)

    # Write standalone verification JSON when verifying generated outputs
    vreport = {'ok': ok, 'total': ok+bad, 'ad_cv_dR_pct': cv(ad_drs) if ad_drs else None,
               'ad_cv_PDPS_pct': cv(ad_pdps) if ad_drs else None, 'ad_cv_R0_pct': cv(ad_r0s) if ad_drs else None}
    if os.path.abspath(input_dir) == os.path.abspath(OUTPUTS_DIR):
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        with open(os.path.join(OUTPUTS_DIR, 'verification_report.json'), 'w') as f:
            json.dump(vreport, f, indent=2)
    return bad == 0


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='all', choices=['forward','aggregate','refine','adfd','verify','all'])
    parser.add_argument('--seed-set', default='paper', choices=['paper','all'],
                        help='paper=4 primary seeds, all=6 seeds (forward always uses all 6)')
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--generated', action='store_true',
                        help='Verify locally generated outputs instead of the bundled outputs')
    args = parser.parse_args()
    if args.phase != 'verify' or args.generated:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)

    seeds = ALL_SEEDS  # forward always uses all 6
    stat_seeds = PRIMARY_SEEDS if args.seed_set == 'paper' else ALL_SEEDS

    log_event(f"=== reproduce_euv.py PHASE={args.phase} DEVICE={DEV} seed_set={args.seed_set} ===")

    if args.phase in ('forward', 'all'):
        for arch in ARCHS:
            for phase in PHASES:
                for seed in seeds:
                    run_forward(arch, phase, seed, skip_if_exists=args.skip_existing)

    if args.phase in ('aggregate', 'all'):
        run_aggregate(seeds=stat_seeds)

    if args.phase in ('refine', 'all'):
        for seed in seeds:
            run_refine(seed)
            torch.cuda.empty_cache()

    if args.phase in ('adfd', 'all'):
        run_adfd()

    if args.phase in ('verify', 'all'):
        input_dir = OUTPUTS_DIR if args.generated or args.phase == 'all' else REFERENCE_DIR
        if not run_verify(input_dir, args.seed_set):
            return 1

    log_event("=== DONE ===")


if __name__ == '__main__':
    sys.exit(main())
