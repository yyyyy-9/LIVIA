import sys
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import fmmax
from fmmax import basis, scattering

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "01_core_operators"))

from python_fmmax.fmmax_adapter_1d import (
    adapt_port_fields,
    demo_mldg,
    film_smatrix_1d,
    order_table,
    solve_patterned,
    solve_uniform,
)
from python_fmmax.fmmax_2d import Engine, compare, rectangle_grid
from python_fmmax.fmmax_phase_shift import shift_layer_solve_result, shift_scattering_matrix
from python_fmmax.order_reader import fmmax_analytic_port


def check_1d_bridges():
    wavelength, period, num_orders, n_grid = 1.065, 0.9, 20, 1024
    n_air, n_sio2, n_ta2o5, n_sub = 1.0, 1.46, 2.14, 1.51
    theta = np.arcsin(wavelength / (2 * period))
    k0 = 2 * np.pi / wavelength

    coeffs, _ = order_table(num_orders)
    expansion = basis.Expansion(np.array(coeffs, dtype=int))
    primitive = fmmax.LatticeVectors(
        u=jnp.asarray([period, 0.0], dtype=jnp.float64),
        v=jnp.asarray([0.0, 1.0], dtype=jnp.float64),
    )
    kxy = jnp.asarray([k0 * np.sin(theta), 0.0], dtype=jnp.float64)
    x = (np.arange(n_grid) + 0.5) / n_grid * period - period / 2.0
    eps_g1 = np.where(np.abs(x) < 0.24, n_sio2**2, n_air**2)[:, None]
    eps_g2 = np.where(np.abs(x) < 0.18, n_ta2o5**2, n_sio2**2)[:, None]

    s_air = solve_uniform(n_air, wavelength, kxy, primitive, expansion)
    s_sio2 = solve_uniform(n_sio2, wavelength, kxy, primitive, expansion)
    s_ta2o5 = solve_uniform(n_ta2o5, wavelength, kxy, primitive, expansion)
    s_sub = solve_uniform(n_sub, wavelength, kxy, primitive, expansion)
    s_g1 = solve_patterned(eps_g1, wavelength, kxy, primitive, expansion)
    s_g2 = solve_patterned(eps_g2, wavelength, kxy, primitive, expansion)

    d_int = np.array([0.18, 0.12] * 3)
    n_int = np.array([n_sio2] + [n_ta2o5, n_sio2] * 3 + [n_sio2])
    orders = np.array([c[0] for c in coeffs])
    kx = k0 * (np.sin(theta) + orders * wavelength / period)

    G1 = scattering.stack_s_matrix(
        [s_air, s_g1, s_sio2], [jnp.asarray(0.0), jnp.asarray(0.35), jnp.asarray(0.0)], force_x64_solve=True
    )
    Sf = film_smatrix_1d(wavelength, kx, d_int, n_int, s_sio2, s_sio2)
    G2 = scattering.stack_s_matrix(
        [s_sio2, s_g2, s_sub], [jnp.asarray(0.0), jnp.asarray(0.40), jnp.asarray(0.0)], force_x64_solve=True
    )
    S_hyb = scattering.redheffer_star_product(
        scattering.redheffer_star_product(G1, Sf, force_x64_solve=True), G2, force_x64_solve=True
    )
    S_nat = scattering.stack_s_matrix(
        [s_air, s_g1, *([s_ta2o5, s_sio2] * 3), s_g2, s_sub],
        [jnp.asarray(0.0), jnp.asarray(0.35), *[jnp.asarray(v) for v in d_int], jnp.asarray(0.40), jnp.asarray(0.0)],
        force_x64_solve=True,
    )
    err_vpi = compare(S_hyb, S_nat)["max_abs"]

    # Sampled 1000x1 unequal ports via native field_conversion_matrix (handles Phi != I and q-sign swaps)
    s_sio2_samp = solve_patterned(np.full((1000, 1), n_sio2**2, complex), wavelength, kxy, primitive, expansion)
    s_sub_samp = solve_patterned(np.full((1000, 1), n_sub**2, complex), wavelength, kxy, primitive, expansion)
    G1_samp = scattering.stack_s_matrix(
        [s_air, s_g1, s_sio2_samp], [jnp.asarray(0.0), jnp.asarray(0.35), jnp.asarray(0.0)], force_x64_solve=True
    )
    Sf_unequal = adapt_port_fields(
        film_smatrix_1d(wavelength, kx, d_int, np.array([n_sio2] + [n_ta2o5, n_sio2] * 3 + [n_sub]), s_sio2, s_sub),
        start_port=s_sio2_samp,
        end_port=s_sub_samp,
    )
    S_samp_hyb = scattering.redheffer_star_product(G1_samp, Sf_unequal, force_x64_solve=True)
    S_samp_nat = scattering.stack_s_matrix(
        [s_air, s_g1, *([s_ta2o5, s_sio2] * 3), s_sub_samp],
        [jnp.asarray(0.0), jnp.asarray(0.35), *[jnp.asarray(v) for v in d_int], jnp.asarray(0.0)],
        force_x64_solve=True,
    )
    return err_vpi, compare(S_samp_hyb, S_samp_nat)["max_abs"]


def check_2d_conical_and_port_reader():
    wl, period, N, theta, phi = 0.532, [2.0, 2.0], 2, 30.0, 30.0
    nI, nII, n_inc = 1.0, 1.4607, 0.93878 + 6.4195j
    eng = Engine(N, wl, period)
    kxy = eng.kxy(theta, phi, nI)
    eps_g = rectangle_grid(period, [128, 128], [1.0, 1.0], nI, n_inc)

    s_I = eng.uniform(kxy, nI)
    s_g = eng.solve(kxy, eps_g)
    s_II = eng.uniform(kxy, nII)
    film_d = np.array([0.01, 0.1, 0.1, 0.1])
    film_n = np.array([nII, n_inc, nII, 2.6678, 1.9044, nII], complex)

    S_g = eng.stack((s_I, s_g, s_II), (0.0, 0.6, 0.0))
    S_film, _ = eng.smm(kxy, film_d, film_n, s_II, s_II, phi)
    S_dev = eng.star(S_g, S_film)
    S_nat = eng.stack((s_I, s_g) + tuple(eng.uniform(kxy, n) for n in film_n[1:]), (0.0, 0.6, *film_d, 0.0))
    err_2d_vpi = compare(S_dev, S_nat)["max_abs"]

    J = eng.incident(kxy, nI, phi)
    err_flux = max(
        eng.native_flux_error(np.asarray(S_dev.s21) @ J, s_I, kxy, nI, phi),
        eng.native_flux_error(np.asarray(S_dev.s11) @ J, s_II, kxy, nII, phi),
    )

    # Selective 2x2 submatrix readout via order_reader.Port vs. full-port efficiencies
    top_port = fmmax_analytic_port(s_I, eng.orders, period, kxy, wl)
    src = top_port.source((0, 0), phi_deg=phi)
    q_m10 = top_port.read(S_dev.s21, src, (-1, 0), direction="up", phi_deg=phi)
    idx_m10 = top_port.index((-1, 0))
    eta_full = eng.efficiencies(S_dev, kxy, nI, nII, phi)["Rtop"][idx_m10]
    err_reader = float(np.max(np.abs(q_m10["eta"] - eta_full)))

    # Unequal ports: analytic (Phi = I) and sampled 50x50 port via adapt_port_fields
    s_II_samp = eng.solve(kxy, np.full((50, 50), nII**2, complex))
    S_unequal, _ = eng.smm(kxy, film_d, np.array([nI, n_inc, nII, 2.6678, 1.9044, nII], complex), s_I, s_II, phi)
    S_samp = adapt_port_fields(S_unequal, end_port=s_II_samp)
    S_nat_unequal = eng.stack(
        (s_I, eng.uniform(kxy, n_inc), s_II, eng.uniform(kxy, 2.6678), eng.uniform(kxy, 1.9044), s_II),
        (0.0, *film_d, 0.0),
    )
    S_nat_samp = eng.stack(
        (s_I, eng.uniform(kxy, n_inc), s_II, eng.uniform(kxy, 2.6678), eng.uniform(kxy, 1.9044), s_II_samp),
        (0.0, *film_d, 0.0),
    )
    err_unequal = max(compare(S_unequal, S_nat_unequal)["max_abs"], compare(S_samp, S_nat_samp)["max_abs"])
    return err_2d_vpi, err_flux, err_reader, err_unequal


def check_two_level_shift():
    wavelength, period, num_orders, n_grid = 1.61, 0.9, 8, 384
    n_ext, n_liq, n_si = 1.45, 1.331, 3.47
    h_slice, cavity = 0.2 / 4, 0.3

    coeffs, _ = order_table(num_orders)
    expansion = basis.Expansion(np.array(coeffs, dtype=int))
    primitive = fmmax.LatticeVectors(
        u=jnp.asarray([period, 0.0], dtype=jnp.float64),
        v=jnp.asarray([0.0, 1.0], dtype=jnp.float64),
    )
    kxy = jnp.asarray([0.0, 0.0], dtype=jnp.float64)

    x = (np.arange(n_grid) + 0.5) / n_grid * period - period / 2.0
    eps_seed = np.where(np.abs(x) < (period / 3.0) / 2.0, n_si**2, n_liq**2)[:, None].astype(np.complex128)

    s_ext = solve_uniform(n_ext, wavelength, kxy, primitive, expansion)
    s_liq = solve_uniform(n_liq, wavelength, kxy, primitive, expansion)
    seed_mode = solve_patterned(eps_seed, wavelength, kxy, primitive, expansion)

    pix_top, pix_bot, pix_offset = np.array([0, 4, 8, 12]), np.array([12, 8, 4, 0]), 24
    dx_top = pix_top * (period / n_grid)
    dx_bot = pix_bot * (period / n_grid)
    dx_macro = pix_offset * (period / n_grid)

    top_modes = [shift_layer_solve_result(seed_mode, float(d)) for d in dx_top]
    bot_modes = [shift_layer_solve_result(seed_mode, float(d)) for d in dx_bot]

    S_upper = scattering.stack_s_matrix(
        [s_ext, *top_modes, s_liq],
        [jnp.asarray(0.0), *[jnp.asarray(h_slice)] * 4, jnp.asarray(cavity)],
        force_x64_solve=True,
    )
    S_lower_0 = scattering.stack_s_matrix(
        [s_liq, *bot_modes, s_ext],
        [jnp.asarray(0.0), *[jnp.asarray(h_slice)] * 4, jnp.asarray(0.0)],
        force_x64_solve=True,
    )
    S_reuse = scattering.redheffer_star_product(
        S_upper, shift_scattering_matrix(S_lower_0, jnp.asarray(dx_macro)), force_x64_solve=True
    )

    def native_slice(pix):
        return solve_patterned(np.roll(eps_seed, int(pix), axis=0), wavelength, kxy, primitive, expansion)

    S_native = scattering.stack_s_matrix(
        [s_ext, *[native_slice(p) for p in pix_top], s_liq, *[native_slice(p + pix_offset) for p in pix_bot], s_ext],
        [jnp.asarray(0.0), *[jnp.asarray(h_slice)] * 4, jnp.asarray(cavity), *[jnp.asarray(h_slice)] * 4, jnp.asarray(0.0)],
        force_x64_solve=True,
    )
    return compare(S_reuse, S_native)["relative_frobenius"]


def main():
    print("FMMAX bridge, port reader & two-level shift verification:")

    ret_ref = {
        "TE": {-1: 0.996499272602, 0: 0.003439597153},
        "TM": {-1: 0.976594231194, 0: 0.022103931915},
    }
    res = demo_mldg(num_orders=40, n_grid=5000)
    max_diff = max(abs(res[pol][m] - ret_ref[pol][m]) for pol in ("TE", "TM") for m in (-1, 0))
    err_int, err_samp = check_1d_bridges()
    print(
        f"  1D: MLDG |d_eta| = {max_diff:.2e} | Interleaved VPI max|dS| = {err_int:.2e} | "
        f"Sampled-port bridge max|dS| = {err_samp:.2e}"
    )
    assert max_diff < 2e-7 and err_int < 1e-11 and err_samp < 1e-11

    err_2d_vpi, err_flux, err_reader, err_unequal = check_2d_conical_and_port_reader()
    print(
        f"  2D: VPI max|dS| = {err_2d_vpi:.2e} | Unequal/sampled port max|dS| = {err_unequal:.2e} | "
        f"Flux |d_eta| = {err_flux:.2e} | Port reader |d_eta| = {err_reader:.2e}"
    )
    assert err_2d_vpi < 1e-10 and err_unequal < 1e-10 and err_flux < 1e-12 and err_reader < 1e-12

    rel_frob = check_two_level_shift()
    print(f"  Double-grating two-level shift (relative Frobenius): {rel_frob:.2e}")
    assert rel_frob < 1e-10


if __name__ == "__main__":
    main()
