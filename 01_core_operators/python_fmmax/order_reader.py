from dataclasses import dataclass
import numpy as np


def flux(f):
    """Signed z-Poynting flux for tangential field vectors [Ex, Ey, Hx, Hy]."""
    return 0.5 * np.real(
        f[..., 0, :] * np.conj(f[..., 3, :]) - f[..., 1, :] * np.conj(f[..., 2, :])
    )


@dataclass
class Source:
    mode_ids: np.ndarray
    amplitudes: np.ndarray
    fields: np.ndarray
    signed_power: np.ndarray
    direction: str
    order: tuple


class Port:
    """Lightweight per-order 4x2 field cache for selective S-submatrix readout."""

    def __init__(self, orders, transverse_k, mode_ids, down, up, description=""):
        self.orders = np.asarray(orders, int).reshape(-1, 2)
        self.transverse_k = np.asarray(transverse_k, float).reshape(-1, 2)
        self.mode_ids = np.asarray(mode_ids, int).reshape(-1, 2)
        self.down = np.asarray(down, complex)
        self.up = np.asarray(up, complex)
        self.description = description

        M = len(self.orders)
        assert self.down.shape == self.up.shape == (M, 4, 2)
        assert self.transverse_k.shape == self.mode_ids.shape == (M, 2)
        assert len(set(map(tuple, self.orders))) == M
        self.lookup = {tuple(o): i for i, o in enumerate(self.orders)}

    def index(self, order):
        key = tuple(order)
        if key not in self.lookup:
            raise KeyError(f"Order {key} is not in the retained expansion.")
        return self.lookup[key]

    def axes(self, order, phi_deg=0.0):
        k = self.transverse_k[self.index(order)]
        norm_k = np.linalg.norm(k)
        if norm_k > 1e-12:
            er = k / norm_k
        else:
            phi_rad = np.deg2rad(phi_deg)
            er = np.array([np.cos(phi_rad), np.sin(phi_rad)])
        es = np.array([-er[1], er[0]])
        return er, es

    def source(self, order=(0, 0), phi_deg=0.0, direction="down"):
        """Construct unit-power TE and TM incident columns for a propagating order."""
        idx = self.index(order)
        field_map = getattr(self, direction)[idx]
        er, es = self.axes(order, phi_deg)

        amp = np.linalg.solve(field_map[:2], np.column_stack((es, er)))
        f_in = field_map @ amp
        p_in = flux(f_in)
        if np.any(np.abs(p_in) < 1e-12) or np.sign(p_in[0]) != np.sign(p_in[1]):
            raise ValueError("Incident order must be propagating.")

        amp = amp / np.sqrt(np.abs(p_in))[None, :]
        f_in = field_map @ amp
        return Source(self.mode_ids[idx], amp, f_in, flux(f_in), direction, tuple(order))

    def read(self, block, source, order, direction="up", phi_deg=0.0):
        """Slice only the 2x2 submatrix of an S-block and compute order efficiency."""
        idx = self.index(order)
        ids = self.mode_ids[idx]
        sub_block = np.asarray(block[..., ids[:, None], source.mode_ids])
        amp = sub_block @ source.amplitudes
        f_out = getattr(self, direction)[idx] @ amp
        return self._readout(f_out, source, order, direction, phi_deg)

    def read_response(self, response, source, order, direction="up", phi_deg=0.0):
        """Read a pre-multiplied modal response column array (e.g. R @ J)."""
        idx = self.index(order)
        ids = self.mode_ids[idx]
        amp = np.asarray(response[..., ids, :])
        f_out = getattr(self, direction)[idx] @ amp
        return self._readout(f_out, source, order, direction, phi_deg)

    def _readout(self, f_out, source, order, direction, phi_deg):
        idx = self.index(order)
        ids = self.mode_ids[idx]
        dir_sign = 1 if direction == source.direction else -1
        eta = dir_sign * flux(f_out) / source.signed_power

        er, es = self.axes(order, phi_deg)
        e_local = np.stack((es @ f_out[..., :2, :], er @ f_out[..., :2, :]), axis=-2)
        h_local = np.stack((er @ f_out[..., 2:, :], es @ f_out[..., 2:, :]), axis=-2)
        eta_te = dir_sign * (-0.5 * np.real(e_local[..., 0, :] * np.conj(h_local[..., 0, :]))) / source.signed_power
        eta_tm = dir_sign * ( 0.5 * np.real(e_local[..., 1, :] * np.conj(h_local[..., 1, :]))) / source.signed_power

        return {
            "order": tuple(order),
            "mode_ids": ids,
            "fields": f_out,
            "eta": eta,
            "eta_TE_out": eta_te,
            "eta_TM_out": eta_tm,
            "electric_tangential_local": e_local,
            "source": source,
            "direction_factor": dir_sign,
        }

    @staticmethod
    def coherent(readout, coefficients):
        """Coherently superpose TE/TM fields for arbitrary Jones polarization."""
        weights = np.asarray(coefficients, complex).reshape(2, 1)
        f_out = readout["fields"] @ weights
        f_inc = readout["source"].fields @ weights
        eta = readout["direction_factor"] * flux(f_out) / flux(f_inc)
        return {"fields": f_out, "eta": eta[..., 0]}

    def save(self, path):
        np.savez_compressed(
            path,
            orders=self.orders,
            transverse_k=self.transverse_k,
            mode_ids=self.mode_ids,
            fields_down=self.down,
            fields_up=self.up,
        )

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls(
                data["orders"],
                data["transverse_k"],
                data["mode_ids"],
                data["fields_down"],
                data["fields_up"],
                "saved port cache",
            )


def fmmax_analytic_port(layer, orders, period, k_parallel, wavelength):
    """Build a Port cache from an FMMAX (1, 1) analytic uniform LayerSolveResult."""
    import jax.numpy as jnp

    orders = np.asarray(orders, int)
    M = len(orders)
    D = 2 * M
    phi = np.asarray(layer.eigenvectors)
    if phi.shape != (D, D) or np.count_nonzero(phi) != D or not np.all(np.diag(phi) == 1):
        raise ValueError("fmmax_analytic_port requires an analytic (1, 1) uniform port.")

    ids = np.column_stack((np.arange(M), np.arange(M) + M))
    ji = jnp.asarray(ids)
    omega = np.asarray(layer.omega_script_k_matrix[ji[:, :, None], ji[:, None, :]])
    q = np.asarray(layer.eigenvalues[ji])
    A = omega / ((2 * np.pi / wavelength) * q[:, None, :])

    down = np.zeros((M, 4, 2), complex)
    down[:, 0, :] = A[:, 1, :]
    down[:, 1, :] = -A[:, 0, :]
    down[:, 2, 0] = 1.0
    down[:, 3, 1] = 1.0

    up = down.copy()
    up[:, :2, :] *= -1
    k_xy = np.asarray(k_parallel)[None, :] + orders * 2 * np.pi / np.asarray(period)[None, :]
    return Port(orders, k_xy, ids, down, up, "FMMAX analytic Hx/Hy amplitudes")
