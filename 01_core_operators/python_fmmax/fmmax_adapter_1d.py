import sys
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import fmmax
from fmmax import basis, fields, fmm, scattering

_CORE_ROOT = Path(__file__).resolve().parent.parent
if str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))
from python_torch_smm.tensor_smm_numpy import film_smm


def order_table(num_orders):
    """Return 1D Fourier order pairs [(0,0), (+1,0), (-1,0), ...] and index map."""
    coeffs = [(0, 0)] + [c for m in range(1, num_orders + 1) for c in ((m, 0), (-m, 0))]
    k_of = {0: 0}
    for m in range(1, num_orders + 1):
        k_of[m] = 2 * m - 1
        k_of[-m] = 2 * m
    return coeffs, k_of


def solve_uniform(n, wavelength, kxy, primitive, expansion, grid=None):
    """Solve a homogeneous isotropic layer via FMMAX's (1, 1) analytic branch (Phi = I)."""
    eps = jnp.asarray([[complex(n) ** 2]])
    return fmm.eigensolve_isotropic_media(jnp.asarray(wavelength), kxy, primitive, eps, expansion)


def solve_patterned(eps_grid, wavelength, kxy, primitive, expansion):
    """Solve a 1D/2D patterned layer on a spatial permittivity grid."""
    return fmm.eigensolve_isotropic_media(
        jnp.asarray(wavelength), kxy, primitive, jnp.asarray(eps_grid, complex), expansion
    )


def star_ab(a, b):
    """Redheffer star product in FMMAX block order (T_down, R_bot, R_top, T_up)."""
    a11, a12, a21, a22 = a
    b11, b12, b21, b22 = b
    eye = np.eye(a11.shape[0])
    s11 = b11 @ np.linalg.solve(eye - a12 @ b21, a11)
    s12 = b12 + b11 @ np.linalg.solve(eye - a12 @ b21, a12 @ b22)
    s21 = a21 + a22 @ np.linalg.solve(eye - b21 @ a12, b21 @ a11)
    s22 = a22 @ np.linalg.solve(eye - b21 @ a12, b22)
    return s11, s12, s21, s22


def film_to_mode_basis(f11, f12, f21, f22, pol, V_top=None, V_bot=None):
    """Convert order-basis SMM blocks to FMMAX H-amplitude layout (s11, s12, s21, s22)."""
    if pol == "TE":
        s11, s12, s21, s22 = f12, -f22, -f11, f21
    elif pol == "TM":
        s11, s12, s21, s22 = f21, f22, f11, f12
    else:
        raise ValueError(f"Unsupported polarization: {pol}")

    if V_top is None and V_bot is None:
        return s11, s12, s21, s22

    Vt = np.eye(s11.shape[0]) if V_top is None else np.asarray(V_top)
    Vb = np.eye(s11.shape[0]) if V_bot is None else np.asarray(V_bot)
    return (
        np.linalg.solve(Vb, s11 @ Vt),
        np.linalg.solve(Vb, s12 @ Vb),
        np.linalg.solve(Vt, s21 @ Vt),
        np.linalg.solve(Vt, s22 @ Vb),
    )


def adapt_port_fields(s_matrix, start_port=None, end_port=None):
    """Map ScatteringMatrix ports via FMMAX native field_conversion_matrix (handles a/b swaps)."""
    td, rb, rt, tu = (np.asarray(getattr(s_matrix, k)) for k in ("s11", "s12", "s21", "s22"))
    s_top = s_matrix.start_layer_solve_result
    s_bot = s_matrix.end_layer_solve_result

    if start_port is not None:
        C0 = np.asarray(fields.field_conversion_matrix(s_top))
        C1 = np.asarray(fields.field_conversion_matrix(start_port))
        K = np.linalg.solve(C0, C1)
        half = len(K) // 2
        U, V, W, X = K[:half, :half], K[:half, half:], K[half:, :half], K[half:, half:]
        rt_new = np.linalg.solve(X - rt @ V, rt @ U - W)
        tu_new = np.linalg.solve(X - rt @ V, tu)
        rb = rb + td @ V @ tu_new
        td = td @ (U + V @ rt_new)
        rt, tu, s_top = rt_new, tu_new, start_port

    if end_port is not None:
        C0 = np.asarray(fields.field_conversion_matrix(s_bot))
        C1 = np.asarray(fields.field_conversion_matrix(end_port))
        K = np.linalg.solve(C0, C1)
        half = len(K) // 2
        U, V, W, X = K[:half, :half], K[:half, half:], K[half:, :half], K[half:, half:]
        td_new = np.linalg.solve(U - rb @ W, td)
        rb_new = np.linalg.solve(U - rb @ W, rb @ X - V)
        rt = rt + tu @ W @ td_new
        tu = tu @ (X + W @ rb_new)
        td, rb, s_bot = td_new, rb_new, end_port

    return scattering.ScatteringMatrix(
        s11=jnp.asarray(td),
        s12=jnp.asarray(rb),
        s21=jnp.asarray(rt),
        s22=jnp.asarray(tu),
        start_layer_solve_result=s_top,
        start_layer_thickness=s_matrix.start_layer_thickness,
        end_layer_solve_result=s_bot,
        end_layer_thickness=s_matrix.end_layer_thickness,
    )


def film_blocks(kx_vector, wavelength, thicknesses, n_list, pol):
    """Evaluate diagonal 1D SMM blocks for a single thickness profile."""
    d = np.asarray(thicknesses, dtype=float).reshape(1, -1)
    out = film_smm(np.asarray(kx_vector), wavelength, d, np.asarray(n_list), pol)
    return tuple(np.diag(np.asarray(v).reshape(-1)) for v in out)


def film_smatrix_1d(wavelength, kx, thicknesses, n_list, s_top, s_bot):
    """Assemble a dual-polarization 1D FMMAX ScatteringMatrix on analytic uniform ports."""
    n = len(kx)
    blocks = [np.zeros((2 * n, 2 * n), complex) for _ in range(4)]
    for pol, ch in (("TE", 0), ("TM", 1)):
        sl = slice(ch * n, (ch + 1) * n)
        mapped = film_to_mode_basis(*film_blocks(kx, wavelength, thicknesses, n_list, pol), pol)
        for b, mat in enumerate(mapped):
            blocks[b][sl, sl] = mat

    return fmmax.ScatteringMatrix(
        s11=jnp.asarray(blocks[0]),
        s12=jnp.asarray(blocks[1]),
        s21=jnp.asarray(blocks[2]),
        s22=jnp.asarray(blocks[3]),
        start_layer_solve_result=s_top,
        start_layer_thickness=jnp.asarray(0.0),
        end_layer_solve_result=s_bot,
        end_layer_thickness=jnp.asarray(0.0),
    )


def demo_mldg(wavelength=1.065, period=0.9, num_orders=40, n_grid=5000):
    """Single-point MLDG hybrid cascade (FMMAX grating + SMM 38-layer mirror via VPI)."""
    n_air, n_sio2, n_ta2o5, n_sub = 1.0, 1.46, 2.14, 1.51
    hg, fill = 1.0, 0.54
    theta = np.arcsin(wavelength / (2 * period))
    k0 = 2 * np.pi / wavelength

    hc = 1.294 - hg
    d = np.array(
        [hc, 0.1003, 0.2687, 0.1353, 0.2000, 0.1878, 0.2226, 0.1212, 0.3479, 0.1295]
        + [0.1995, 0.1295] * 14
    )
    n_list = np.array([n_sub] + [n_sio2, n_ta2o5] * 19 + [n_sub])

    coeffs, k_of = order_table(num_orders)
    expansion = basis.Expansion(np.array(coeffs, dtype=int))
    primitive = fmmax.LatticeVectors(
        u=jnp.asarray([period, 0.0], dtype=jnp.float64),
        v=jnp.asarray([0.0, 1.0], dtype=jnp.float64),
    )
    kxy = jnp.asarray([k0 * np.sin(theta), 0.0], dtype=jnp.float64)

    x = (np.arange(n_grid) + 0.5) / n_grid * period - period / 2
    eps_g = np.where(np.abs(x) < period * fill / 2, n_sio2**2, n_air**2)[:, None]

    s_air = solve_uniform(n_air, wavelength, kxy, primitive, expansion)
    s_g = solve_patterned(eps_g, wavelength, kxy, primitive, expansion)
    s_sub = solve_uniform(n_sub, wavelength, kxy, primitive, expansion)

    Sg = scattering.stack_s_matrix(
        [s_air, s_g, s_sub],
        [jnp.asarray(0.0), jnp.asarray(hg), jnp.asarray(0.0)],
        force_x64_solve=True,
    )

    n = expansion.num_terms
    orders = np.array([c[0] for c in coeffs])
    kx = k0 * (np.sin(theta) + orders * wavelength / period)
    Sf = film_smatrix_1d(wavelength, kx, d, n_list, s_sub, s_sub)
    Sc = scattering.redheffer_star_product(Sg, Sf, force_x64_solve=True)
    s21_out = np.asarray(Sc.s21)

    result = {}
    for pol, ch in (("TE", 0), ("TM", 1)):
        result[pol] = {
            m: float(abs(s21_out[ch * n + k_of[m], ch * n + k_of[0]]) ** 2) for m in (-1, 0)
        }
    return result
