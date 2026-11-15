import numpy as np


def film_smm(kx, wavelength, d, n_list, pol):
    """Channel-decoupled batch S-matrix recursion for planar multilayer stacks."""
    kx = np.asarray(kx, dtype=np.complex128).reshape(-1)
    d = np.asarray(d, dtype=np.float64)
    if d.ndim == 1:
        d = d.reshape(1, -1)
    n_list = np.asarray(n_list, dtype=np.complex128).reshape(-1)

    k0 = 2.0 * np.pi / float(wavelength)
    kz = np.sqrt((n_list[:, None] * k0) ** 2 - (kx[None, :] + 1e-16j) ** 2)
    kz[np.imag(kz) < -1e-14] *= -1
    kz[(kz.real < 0) & (np.abs(kz.imag) < 1e-14)] *= -1

    shape = (d.shape[0], kx.shape[0])
    s11 = np.zeros(shape, dtype=np.complex128)
    s12 = np.ones(shape, dtype=np.complex128)
    s21 = np.ones(shape, dtype=np.complex128)
    s22 = np.zeros(shape, dtype=np.complex128)

    is_te = pol.upper() in ("TE", "S")
    for j in range(d.shape[1], -1, -1):
        ku, kl = kz[j], kz[j + 1]
        if is_te:
            denom = ku + kl
            r = (ku - kl) / denom
            td = 2.0 * ku / denom
            tu = 2.0 * kl / denom
        else:
            eu, el = n_list[j] ** 2, n_list[j + 1] ** 2
            denom = el * ku + eu * kl
            r = (el * ku - eu * kl) / denom
            td = 2.0 * el * ku / denom
            tu = 2.0 * eu * kl / denom

        if j < d.shape[1]:
            phase = np.exp(1j * d[:, j, None] * kl[None, :])
            s11 = s11 * (phase ** 2)
            s12 = s12 * phase
            s21 = s21 * phase

        inv_den = 1.0 / (1.0 + r * s11)
        s11, s12, s21, s22 = (
            r + tu * s11 * inv_den * td,
            tu * inv_den * s12,
            s21 * inv_den * td,
            s22 - s21 * inv_den * r * s12,
        )
    return s11, s12, s21, s22


def film_native(raw_blocks, pol):
    """Align standard SMM blocks (R_top, T_up, T_down, R_bot) to FMMAX H-amplitudes."""
    f11, f12, f21, f22 = raw_blocks
    if pol.upper() in ("TE", "S"):
        return -f11, f21, f12, -f22
    return f11, f12, f21, f22


def batched_cascade_grating_film(grating_blocks, film_diag_blocks):
    """Four-block Redheffer star product of a dense grating S and batched diagonal film S."""
    a11, a12, a21, a22 = [np.asarray(x, dtype=np.complex128) for x in grating_blocks]
    b11, b12, b21, b22 = [np.asarray(x, dtype=np.complex128) for x in film_diag_blocks]

    eye = np.eye(a11.shape[-1], dtype=np.complex128)
    z = np.linalg.inv(eye - a22[None, :, :] * b11[:, None, :])
    zp = np.linalg.inv(eye - b11[:, :, None] * a22[None, :, :])

    s11 = a11[None] + a12[None] @ ((b11[:, :, None] * z) @ a21[None])
    s21 = b21[:, :, None] * (z @ a21[None])
    zp_b12 = zp * b12[:, None, :]
    s12 = a12[None] @ zp_b12
    s22 = b21[:, :, None] * (a22[None] @ zp_b12)
    diag_idx = np.arange(a11.shape[-1])
    s22[:, diag_idx, diag_idx] += b22
    return s11, s12, s21, s22
