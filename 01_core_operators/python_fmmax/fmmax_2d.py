from dataclasses import dataclass
from pathlib import Path
import re
import sys

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from fmmax import basis, fields, fmm, scattering
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parent
CORE_ROOT = ROOT.parent
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))
from python_torch_smm.tensor_smm_numpy import film_native, film_smm

KEYS = ("s11", "s12", "s21", "s22")


def square_orders(N):
    """Build the (2N + 1)^2 square 2D Fourier order table."""
    mx, my = np.meshgrid(np.arange(-N, N + 1), np.arange(-N, N + 1), indexing="xy")
    return np.column_stack((mx.ravel(), my.ravel())).astype(int)


def material_from_matlab(name, wavelength):
    """Interpolate n + ik at wavelength (um) from get_<name>_nk.m."""
    path = CORE_ROOT / "matlab_reticolo" / f"get_{name}_nk.m"
    txt = path.read_text(encoding="utf-8")
    raw = re.search(r"raw_data\s*=\s*\[([\s\S]*?)\];", txt).group(1)
    rows = np.array([[float(x) for x in line.split()] for line in raw.splitlines() if line.strip()])
    n = float(PchipInterpolator(rows[:, 0], rows[:, 1])(wavelength))
    k = float(PchipInterpolator(rows[:, 0], rows[:, 2])(wavelength)) if rows.shape[1] > 2 else 0.0
    return complex(n, k)


def rectangle_grid(period, grid, width, background, index, center=(0.0, 0.0)):
    """Sample a 2D rectangular pillar permittivity grid on a periodic cell."""
    px, py = np.broadcast_to(period, 2)
    gx, gy = np.broadcast_to(grid, 2).astype(int)
    xx = (np.arange(gx) + 0.5) * px / gx - px / 2
    yy = (np.arange(gy) + 0.5) * py / gy - py / 2
    dx = (xx - center[0] + px / 2) % px - px / 2
    dy = (yy - center[1] + py / 2) % py - py / 2
    inside = (np.abs(dx[:, None]) < width[0] / 2) & (np.abs(dy[None, :]) < width[1] / 2)
    return np.where(inside, complex(index) ** 2, complex(background) ** 2).astype(np.complex128)


class Engine:
    """2D conical FMMAX wrapper with local-(s, p) to global-(Hx, Hy) SMM bridging."""

    def __init__(self, N, wavelength, period, formulation="JONES_DIRECT_FOURIER"):
        self.N = N
        self.wavelength = float(wavelength)
        self.period = np.broadcast_to(period, 2).astype(float)
        self.orders = square_orders(N)
        self.M = len(self.orders)
        self.D = 2 * self.M
        self.zero = int(np.flatnonzero(np.all(self.orders == 0, axis=1))[0])
        self.expansion = basis.Expansion(self.orders)
        self.formulation = formulation
        self.lattice = basis.LatticeVectors(
            u=jnp.array([self.period[0], 0.0], dtype=jnp.float64),
            v=jnp.array([0.0, self.period[1]], dtype=jnp.float64),
        )
        self._solve = jax.jit(
            lambda kxy, eps: fmm.eigensolve_isotropic_media(
                jnp.asarray(self.wavelength),
                kxy,
                self.lattice,
                eps,
                self.expansion,
                formulation=getattr(fmm.Formulation, self.formulation),
            )
        )
        self._solve_with_vector = jax.jit(
            lambda kxy, eps, tx, ty: fmm.eigensolve_isotropic_media(
                jnp.asarray(self.wavelength),
                kxy,
                self.lattice,
                eps,
                self.expansion,
                formulation=lambda eps_, exp_, lattice_: (tx, ty),
            )
        )
        self.stack = jax.jit(
            lambda layers, ds: scattering.stack_s_matrix(
                layers,
                tuple(jnp.asarray(d, dtype=jnp.float64) for d in ds),
                force_x64_solve=True,
            )
        )
        self.star = jax.jit(
            lambda a, b: scattering.redheffer_star_product(a, b, force_x64_solve=True)
        )

    def kxy(self, theta, phi, n=1.0):
        th, ph = np.deg2rad([theta, phi])
        k0 = 2 * np.pi / self.wavelength
        return np.array([np.cos(ph), np.sin(ph)]) * float(np.real(n)) * k0 * np.sin(th)

    def solve(self, kxy, eps):
        return self._solve(
            jnp.asarray(kxy, dtype=jnp.float64),
            jnp.asarray(eps, dtype=jnp.complex128),
        )

    def uniform(self, kxy, n):
        return self.solve(kxy, np.array([[complex(n) ** 2]], complex))

    def solve_with_vector(self, kxy, eps, tx, ty):
        """Eigensolve with an explicit co-translated tangent vector field."""
        return self._solve_with_vector(
            jnp.asarray(kxy, dtype=jnp.float64),
            jnp.asarray(eps, dtype=jnp.complex128),
            jnp.asarray(tx),
            jnp.asarray(ty),
        )

    def transverse(self, kxy):
        return np.asarray(kxy)[None, :] + self.orders * 2 * np.pi / self.period[None, :]

    def local_axes(self, kxy, phi=0.0):
        k = self.transverse(kxy)
        rho = np.linalg.norm(k, axis=1)
        c = np.divide(k[:, 0], rho, out=np.full_like(rho, np.cos(np.deg2rad(phi))), where=rho > 1e-12)
        s = np.divide(k[:, 1], rho, out=np.full_like(rho, np.sin(np.deg2rad(phi))), where=rho > 1e-12)
        return c, s, rho

    def make_s(self, blocks, start, end):
        return scattering.ScatteringMatrix(
            **{k: jnp.asarray(v) for k, v in zip(KEYS, blocks)},
            start_layer_solve_result=start,
            start_layer_thickness=jnp.asarray(0.0),
            end_layer_solve_result=end,
            end_layer_thickness=jnp.asarray(0.0),
        )

    def smm(self, kxy, d, n_list, start, end, phi=0.0):
        """Convert 1D radial-wavevector SMM blocks to a 2D FMMAX ScatteringMatrix."""
        c, s, rho = self.local_axes(kxy, phi)
        raw = {
            p: film_smm(rho, self.wavelength, np.asarray(d).reshape(1, -1), np.asarray(n_list), p)
            for p in ("TE", "TM")
        }
        local = {p: film_native(raw[p], p) for p in raw}

        blocks = []
        ii = np.arange(self.M)
        for b in (2, 3, 0, 1):
            a_te = np.asarray(local["TE"][b]).reshape(-1)
            a_tm = np.asarray(local["TM"][b]).reshape(-1)
            mat = np.zeros((self.D, self.D), complex)
            mat[ii, ii] = c * c * a_te + s * s * a_tm
            mat[ii, ii + self.M] = c * s * (a_te - a_tm)
            mat[ii + self.M, ii] = c * s * (a_te - a_tm)
            mat[ii + self.M, ii + self.M] = s * s * a_te + c * c * a_tm
            blocks.append(mat)
        return self.make_s(blocks, start, end), raw

    def incident(self, kxy, n, phi=0.0):
        c, s, _ = self.local_axes(kxy, phi)
        J = np.zeros((self.D, 2), complex)
        i0 = self.zero
        J[i0, :] = [c[i0], -s[i0]]
        J[i0 + self.M, :] = [s[i0], c[i0]]
        weights = self.power(J, kxy, n, phi).sum(axis=0)
        if np.any(weights <= 0):
            raise ValueError("Incident zero-order wave is not propagating.")
        return J / np.sqrt(weights)[None, :]

    def power(self, amplitude, kxy, n, phi=0.0):
        """Per-order Poynting flux on an isotropic analytic uniform port."""
        h = np.asarray(amplitude)
        c, s, rho = self.local_axes(kxy, phi)
        k0 = 2 * np.pi / self.wavelength
        q = np.sqrt((complex(n) * k0) ** 2 - rho ** 2 + 0j)
        q = np.where(q.imag < 0, -q, q)

        h_te = c[:, None] * h[:self.M] + s[:, None] * h[self.M:]
        h_tm = -s[:, None] * h[:self.M] + c[:, None] * h[self.M:]
        w_te = 0.5 * np.real(k0 / q)
        w_tm = 0.5 * np.real(q / (k0 * complex(n) ** 2))
        return w_te[:, None] * np.abs(h_te) ** 2 + w_tm[:, None] * np.abs(h_tm) ** 2

    def native_flux_error(self, amplitude, port, kxy, n, phi=0.0):
        h = jnp.asarray(amplitude)
        fw, _ = fields.amplitude_poynting_flux(h, jnp.zeros_like(h), port)
        fw = np.asarray(fw)
        native = fw[:self.M] + fw[self.M:]
        return float(np.max(np.abs(native - self.power(amplitude, kxy, n, phi))))

    def efficiencies(self, S, kxy, nt, nb, phi=0.0):
        jt = self.incident(kxy, nt, phi)
        jb = self.incident(kxy, nb, phi)
        vals = arrays(S)
        return {
            "Rtop": self.power(vals[2] @ jt, kxy, nt, phi),
            "Tdown": self.power(vals[0] @ jt, kxy, nb, phi),
            "Rbottom": self.power(vals[1] @ jb, kxy, nb, phi),
            "Tup": self.power(vals[3] @ jb, kxy, nt, phi),
        }

    def phase(self, kxy, delta):
        p = np.exp(-1j * (self.transverse(kxy) @ np.asarray(delta)))
        return np.r_[p, p]

    def shift(self, S, kxy, delta):
        p = jnp.asarray(self.phase(kxy, delta))
        return scattering.ScatteringMatrix(
            **{k: p[:, None] * getattr(S, k) * jnp.conj(p[None, :]) for k in KEYS},
            start_layer_solve_result=S.start_layer_solve_result,
            start_layer_thickness=S.start_layer_thickness,
            end_layer_solve_result=S.end_layer_solve_result,
            end_layer_thickness=S.end_layer_thickness,
        )


def arrays(S):
    return tuple(np.asarray(getattr(S, k)) for k in KEYS)


def compare(actual, expected):
    a = arrays(actual) if hasattr(actual, "s11") else actual
    b = arrays(expected) if hasattr(expected, "s11") else expected
    diffs = [float(np.max(np.abs(x - y))) for x, y in zip(a, b)]
    num = np.sqrt(sum(np.linalg.norm(x - y) ** 2 for x, y in zip(a, b)))
    den = max(1.0, np.sqrt(sum(np.linalg.norm(y) ** 2 for y in b)))
    return {
        "max_abs": max(diffs),
        "by_block": dict(zip(KEYS, diffs)),
        "relative_frobenius": float(num / den),
    }


@dataclass
class ReducedTop:
    """Pre-contracted top-incidence blocks for fixed-illumination sweeps."""
    r0: np.ndarray
    t0: np.ndarray
    tup: np.ndarray
    rbottom: np.ndarray

    @classmethod
    def from_full(cls, S, J, top_rows=None):
        td, rb, rt, tu = arrays(S)
        rows = np.arange(rt.shape[0]) if top_rows is None else np.asarray(top_rows)
        return cls(rt[rows] @ J, td @ J, tu[rows], rb)


def reduced_cascade(A: ReducedTop, B, transmission_rows=None, interface=None):
    """Cascade a pre-contracted top subsystem with a lower subsystem."""
    td, _, rt, _ = arrays(B)
    rows = np.arange(td.shape[0]) if transmission_rows is None else np.asarray(transmission_rows)
    ids = np.arange(rt.shape[0]) if interface is None else np.asarray(interface)
    ra = A.rbottom[np.ix_(ids, ids)]
    rb_sub = rt[np.ix_(ids, ids)]
    down = np.linalg.solve(np.eye(len(ids)) - ra @ rb_sub, A.t0[ids])
    reflected = A.r0 + A.tup[:, ids] @ (rb_sub @ down)
    transmitted = td[np.ix_(rows, ids)] @ down
    return reflected, transmitted
