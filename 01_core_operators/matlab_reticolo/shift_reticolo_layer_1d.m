function shifted_layer = shift_reticolo_layer_1d(seed_layer, dx, alpha, unit_scale)
% Translate a 1D RETICOLO layer eigensolution (aa{1}{tex_id}) by dx without EVD.

    if iscell(alpha)
        alpha = alpha{2};
    end
    shifted_layer = seed_layer;
    n_modes = numel(alpha);
    phase = exp(-1i * alpha(:) * (dx / unit_scale));
    P_diag = spdiags(phase, 0, n_modes, n_modes);
    P_inv  = spdiags(conj(phase), 0, n_modes, n_modes);

    shifted_layer{1} = P_diag * seed_layer{1};
    shifted_layer{2} = seed_layer{2} * P_inv;
    shifted_layer{3} = P_diag * seed_layer{3};
    if numel(seed_layer) >= 4 && ~isempty(seed_layer{4})
        shifted_layer{4} = seed_layer{4} * P_inv;
    end
    if numel(seed_layer) >= 7
        if ~isempty(seed_layer{6})
            shifted_layer{6} = P_diag * seed_layer{6};
        end
        if ~isempty(seed_layer{7})
            shifted_layer{7} = P_diag * seed_layer{7};
        end
    end
    if numel(seed_layer) >= 8 && ~isempty(seed_layer{8})
        shifted_layer{8} = shift_wall_descriptor(seed_layer{8}, dx, unit_scale);
    end
end

function w_new = shift_wall_descriptor(w_seed, dx, unit_scale)
    old_coords = w_seed{1};
    old_mats   = w_seed{2};
    period     = old_coords(end);
    dx_norm    = dx / unit_scale;
    walls_new  = sort(mod(old_coords(1:end-1) + dx_norm, period));
    new_coords = [walls_new, period];
    x_starts   = [0, new_coords(1:end-1)];
    x_mids     = 0.5 * (x_starts + new_coords);
    x_query    = mod(x_mids - dx_norm, period);
    indices    = sum(x_query(:) > old_coords(:).', 2) + 1;
    w_new      = {new_coords, old_mats(:, indices)};
end
