function S_shifted = shift_reticolo_smatrix_1d(S_cell, dx, alpha, unit_scale)
% Translate a 1D RETICOLO subsystem S-matrix with uniform ports by dx.

    if iscell(alpha)
        alpha = alpha{2};
    end
    n_modes = size(S_cell{1}, 1) / 2;
    alpha_vec = alpha(1:n_modes);
    phase = exp(-1i * alpha_vec(:) * (dx / unit_scale));
    p_full = [phase; phase];

    S_shifted = S_cell;
    S_shifted{1} = (p_full .* S_cell{1}) .* conj(p_full).';
end
