function out = reticolo_orders(action, varargin)
% Port caching and single-order readout for unfolded 2D RETICOLO S-matrices.
%   p   = reticolo_orders('port', aa, texture_id, sens)
%   src = reticolo_orders('source', p, [0, 0], phi_deg)
%   q   = reticolo_orders('read', R_or_T, p, src, [mx, my], 'up'|'down')
%   c   = reticolo_orders('coherent', q, jones_weights)

    switch action
        case 'port'
            aa = retio(varargin{1});
            tex_id = varargin{2};
            sens = varargin{3};
            assert(aa{2}{end}.dim == 2, 'reticolo_orders is for 2D RETICOLO ports.');

            [~, ~, ~, ~, ~, ray] = retb(aa{2}, aa{1}{tex_id}, sens);
            beta = ray{1}(:, 1:2);
            M = size(beta, 1);
            D = aa{2}{1};
            assert(D == 2 * M, 'Disable symmetry folding before caching a 2D port.');

            field_mat = [ray{2}; ray{3}; ray{4}; ray{5}];
            p.orders = round((beta - aa{6}) ./ (2 * pi * aa{4} ./ aa{5}));
            p.transverse_k = beta / aa{4};
            p.M = M;
            p.D = D;
            p.mode_ids = zeros(M, 2);
            p.down = zeros(4, 2, M);
            p.up = zeros(4, 2, M);

            for i = 1:M
                rows = [i, i + M, i + 2 * M, i + 3 * M];
                ids = find(any(field_mat(rows, :) ~= 0, 1));
                assert(numel(ids) == 2, 'Port modes mix diffraction orders.');
                p.mode_ids(i, :) = ids;
                p.down(:, :, i) = full(field_mat(rows, ids));
                p.up(:, :, i) = p.down(:, :, i);
                p.up(3:4, :, i) = -p.up(3:4, :, i);
            end
            assert(numel(unique(p.mode_ids(:))) == D);
            out = p;

        case 'source'
            p = varargin{1};
            order = varargin{2};
            phi_deg = varargin{3};
            idx = find_order_index(p, order);
            [er, es] = local_axes(p, idx, phi_deg);

            f_down = p.down(:, :, idx);
            amp = f_down(1:2, :) \ [es, er];
            p_in = poynting_z(f_down * amp);
            assert(all(abs(p_in) > 1e-12) && sign(p_in(1)) == sign(p_in(2)), ...
                'Selected incident order is not propagating.');

            amp = amp ./ sqrt(abs(p_in));
            fields_in = f_down * amp;
            out = struct( ...
                'mode_ids', p.mode_ids(idx, :), ...
                'amplitudes', amp, ...
                'fields', fields_in, ...
                'signed_power', poynting_z(fields_in), ...
                'direction', 'down', ...
                'phi', phi_deg, ...
                'order', order);

        case {'read', 'response'}
            B = varargin{1};
            p = varargin{2};
            src = varargin{3};
            order = varargin{4};
            direction = varargin{5};

            idx = find_order_index(p, order);
            ids = p.mode_ids(idx, :);
            if strcmp(action, 'response')
                amp = B(ids, :, :);
            else
                amp = pagemtimes(B(ids, src.mode_ids, :), src.amplitudes);
            end

            f = pagemtimes(p.(direction)(:, :, idx), amp);
            dir_sign = 2 * strcmp(direction, src.direction) - 1;
            eta = dir_sign * poynting_z(f) ./ src.signed_power;

            [er, es] = local_axes(p, idx, src.phi);
            e_loc = pagemtimes([es.'; er.'], f(1:2, :, :));
            h_loc = pagemtimes([er.'; es.'], f(3:4, :, :));
            eta_te = dir_sign * (-0.5 * real(e_loc(1, :, :) .* conj(h_loc(1, :, :)))) ./ src.signed_power;
            eta_tm = dir_sign * ( 0.5 * real(e_loc(2, :, :) .* conj(h_loc(2, :, :)))) ./ src.signed_power;

            out = struct( ...
                'order', order, ...
                'mode_ids', ids, ...
                'fields', f, ...
                'eta', eta, ...
                'eta_TE_out', eta_te, ...
                'eta_TM_out', eta_tm, ...
                'electric_tangential_local', e_loc, ...
                'source', src, ...
                'direction_factor', dir_sign);

        case 'coherent'
            q = varargin{1};
            weights = varargin{2}(:);
            f = pagemtimes(q.fields, weights);
            f_inc = q.source.fields * weights;
            out = struct( ...
                'fields', f, ...
                'eta', q.direction_factor * poynting_z(f) ./ poynting_z(f_inc));

        otherwise
            error('Unknown action: %s', action);
    end
end

function idx = find_order_index(p, order)
    idx = find(all(p.orders == reshape(order, 1, 2), 2));
    assert(isscalar(idx), 'Requested diffraction order is missing or ambiguous.');
end

function [er, es] = local_axes(p, idx, phi_deg)
    k = p.transverse_k(idx, :).';
    if norm(k) > 1e-12
        er = k / norm(k);
    else
        er = [cosd(phi_deg); sind(phi_deg)];
    end
    es = [-er(2); er(1)];
end

function sz = poynting_z(f)
    sz = 0.5 * real(f(1, :, :) .* conj(f(4, :, :)) - f(2, :, :) .* conj(f(3, :, :)));
end
