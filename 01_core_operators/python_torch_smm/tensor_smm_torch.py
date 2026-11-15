import torch


def fast_batch_smatrix_vd_torch(kx_vector, wavelength, d_matrix, n_list, pol):
    """Differentiable batch S-matrix solver for planar multilayer stacks (complex128)."""
    if d_matrix.dim() == 1:
        d_matrix = d_matrix.unsqueeze(0)

    k0 = 2 * torch.pi / wavelength
    num_orders = kx_vector.shape[0]
    num_batches, num_layers = d_matrix.shape
    total_media = len(n_list)
    device = d_matrix.device

    kx_reg = kx_vector + 1e-16j
    n_tensors = [
        n_val.to(device=device, dtype=torch.complex128)
        if torch.is_tensor(n_val)
        else torch.tensor(n_val, dtype=torch.complex128, device=device)
        for n_val in n_list
    ]
    kz_matrix = torch.zeros((total_media, num_orders), dtype=torch.complex128, device=device)
    for i, n_val in enumerate(n_tensors):
        kz_curr = torch.sqrt((n_val * k0) ** 2 - kx_reg ** 2)
        neg_imag = torch.imag(kz_curr) < -1e-14
        kz_curr[neg_imag] = -kz_curr[neg_imag]
        neg_real = (torch.real(kz_curr) < 0) & (torch.abs(torch.imag(kz_curr)) < 1e-14)
        kz_curr[neg_real] = -kz_curr[neg_real]
        kz_matrix[i, :] = kz_curr

    s11 = torch.zeros((num_batches, num_orders), dtype=torch.complex128, device=device)
    s12 = torch.ones((num_batches, num_orders), dtype=torch.complex128, device=device)
    s21 = torch.ones((num_batches, num_orders), dtype=torch.complex128, device=device)
    s22 = torch.zeros((num_batches, num_orders), dtype=torch.complex128, device=device)

    is_te = pol.lower() in ("s", "te")
    for j in range(num_layers + 1, 0, -1):
        kz_up = kz_matrix[j - 1, :]
        kz_lo = kz_matrix[j, :]
        n_up = n_tensors[j - 1]
        n_lo = n_tensors[j]

        if is_te:
            denom = kz_up + kz_lo
            r_loc = (kz_up - kz_lo) / denom
            t_down = 2 * kz_up / denom
            t_up = 2 * kz_lo / denom
        else:
            eps_up = n_up ** 2
            eps_lo = n_lo ** 2
            denom = eps_lo * kz_up + eps_up * kz_lo
            r_loc = (eps_lo * kz_up - eps_up * kz_lo) / denom
            t_down = 2 * eps_lo * kz_up / denom
            t_up = 2 * eps_up * kz_lo / denom

        if j <= num_layers:
            phi = torch.exp(1j * torch.outer(d_matrix[:, j - 1], kz_lo))
            s11 = s11 * (phi ** 2)
            s12 = s12 * phi
            s21 = s21 * phi

        inv_den = 1.0 / (1 + r_loc * s11)
        s11, s12, s21, s22 = (
            r_loc + t_up * s11 * inv_den * t_down,
            t_up * inv_den * s12,
            s21 * inv_den * t_down,
            s22 - s21 * inv_den * r_loc * s12,
        )

    return s11, s12, s21, s22
