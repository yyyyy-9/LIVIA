function [S11, S12, S21, S22] = FastBatch_SMatrix_vd(kx_vector, wavelength, d_matrix, n_list, pol)
% Channel-decoupled batch S-matrix solver for isotropic planar film stacks.
% d_matrix: B-by-L thickness matrix; n_list: [n_top, n_1, ..., n_L, n_bot].

    total_media = numel(n_list);
    expected_layers = total_media - 2;
    if expected_layers > 1 && iscolumn(d_matrix) && numel(d_matrix) == expected_layers
        d_matrix = d_matrix.';
    end
    assert(size(d_matrix, 2) == expected_layers, 'LIVIA:ThicknessShape', ...
        'd_matrix must have numel(n_list) - 2 columns.');

    k0 = 2 * pi / wavelength;
    num_orders = length(kx_vector);
    [num_batches, num_layers] = size(d_matrix);
    kx_vector = kx_vector + 1e-16i;

    kz_matrix = zeros(total_media, num_orders);
    for i = 1:total_media
        kz_curr = sqrt((n_list(i) * k0)^2 - kx_vector.^2);
        neg_imag = imag(kz_curr) < -1e-14;
        kz_curr(neg_imag) = -kz_curr(neg_imag);
        neg_real = (real(kz_curr) < 0) & (abs(imag(kz_curr)) < 1e-14);
        kz_curr(neg_real) = -kz_curr(neg_real);
        kz_matrix(i, :) = kz_curr;
    end

    S11 = zeros(num_batches, num_orders);
    S12 = ones(num_batches, num_orders);
    S21 = ones(num_batches, num_orders);
    S22 = zeros(num_batches, num_orders);

    is_te = strcmpi(pol, 's') || strcmpi(pol, 'TE');
    for j = (num_layers + 1):-1:1
        kz_up = kz_matrix(j, :);
        kz_lo = kz_matrix(j + 1, :);

        if is_te
            denom = kz_up + kz_lo;
            r_loc = (kz_up - kz_lo) ./ denom;
            t_down = 2 * kz_up ./ denom;
            t_up   = 2 * kz_lo ./ denom;
        else
            eps_up = n_list(j)^2;
            eps_lo = n_list(j + 1)^2;
            denom = eps_lo * kz_up + eps_up * kz_lo;
            r_loc = (eps_lo * kz_up - eps_up * kz_lo) ./ denom;
            t_down = 2 * eps_lo * kz_up ./ denom;
            t_up   = 2 * eps_up * kz_lo ./ denom;
        end

        if j <= num_layers
            phi = exp(1i * d_matrix(:, j) * kz_lo);
            S11 = S11 .* (phi.^2);
            S12 = S12 .* phi;
            S21 = S21 .* phi;
        end

        inv_den = 1.0 ./ (1 + r_loc .* S11);
        S11_next = r_loc + t_up .* S11 .* inv_den .* t_down;
        S12_next = t_up .* inv_den .* S12;
        S21_next = S21 .* inv_den .* t_down;
        S22_next = S22 - S21 .* inv_den .* r_loc .* S12;

        S11 = S11_next;
        S12 = S12_next;
        S21 = S21_next;
        S22 = S22_next;
    end
end
