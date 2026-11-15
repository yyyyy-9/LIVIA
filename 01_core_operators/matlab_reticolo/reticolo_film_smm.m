function S = reticolo_film_smm(aa, top_id, bottom_id, wavelength, d, n_list, parm)
% Convert planar SMM blocks to a full RETICOLO modal S-matrix via native retb.
% Supports unequal terminal media; n_list = [n_top, n_1, ..., n_L, n_bot].
% Output follows RETICOLO block order: [T_up, R_top; R_bot, T_down].

    aa = retio(aa);
    init = aa{2};
    D = init{1};
    dim = init{end}.dim;

    [~, ~, ~, ~, ~, rt] = retb(init, aa{1}{top_id}, parm.res2.tolh);
    [~, ~, ~, ~, ~, rb] = retb(init, aa{1}{bottom_id}, -parm.res2.tolb);

    k_xy = init{2}.' / aa{4};
    rho = sqrt(sum(k_xy.^2, 2)).';

    if dim == 1
        if aa{end} == 0
            pol = 'TE';
        else
            pol = 'TM';
        end
        f = cell(1, 4);
        [f{:}] = FastBatch_SMatrix_vd(rho, wavelength, d, n_list, pol);

        Qt = rt{2};
        Qb = rb{2};
        M = numel(rho);
        b = {
            Qt \ (spdiags(f{1}(:), 0, M, M) * Qt), ...
            Qt \ (spdiags(f{2}(:), 0, M, M) * Qb), ...
            Qb \ (spdiags(f{3}(:), 0, M, M) * Qt), ...
            Qb \ (spdiags(f{4}(:), 0, M, M) * Qb)
        };
    else
        M = size(k_xy, 1);
        assert(D == 2 * M, 'Unfolded 2D port modes (D == 2*M) are required.');

        te = cell(1, 4);
        tm = cell(1, 4);
        [te{:}] = FastBatch_SMatrix_vd(rho, wavelength, d, n_list, 'TE');
        [tm{:}] = FastBatch_SMatrix_vd(rho, wavelength, d, n_list, 'TM');

        Qt = [rt{2}; rt{3}];
        Qb = [rb{2}; rb{3}];
        k0 = 2 * pi / wavelength;
        q_top = outgoing_kz((n_list(1) * k0)^2 - rho.^2);
        q_bot = outgoing_kz((n_list(end) * k0)^2 - rho.^2);
        Y_top = q_top / n_list(1)^2;
        Y_bot = q_bot / n_list(end)^2;

        b = repmat({complex(zeros(D))}, 1, 4);
        for j = 1:M
            rows = [j, j + M];
            it = find(any(Qt(rows, :) ~= 0, 1));
            ib = find(any(Qb(rows, :) ~= 0, 1));
            assert(numel(it) == 2 && numel(ib) == 2, 'Unexpected uniform port support.');

            if rho(j) > 1e-12
                er = k_xy(j, :).' / rho(j);
            else
                er = [1; 0];
            end
            es = [-er(2); er(1)];
            U = [es, er];

            % Convert TM magnetic amplitudes to tangential electric amplitudes
            tm_e = [
                -tm{1}(j), ...
                (Y_top(j) / Y_bot(j)) * tm{2}(j), ...
                (Y_bot(j) / Y_top(j)) * tm{3}(j), ...
                -tm{4}(j)
            ];
            src = {Qt(rows, it), Qb(rows, ib), Qt(rows, it), Qb(rows, ib)};
            dst = {Qt(rows, it), Qt(rows, it), Qb(rows, ib), Qb(rows, ib)};
            out_ids = {it, it, ib, ib};
            in_ids  = {it, ib, it, ib};

            for z = 1:4
                B_local = U * diag([te{z}(j), tm_e(z)]) * U.';
                b{z}(out_ids{z}, in_ids{z}) = dst{z} \ (B_local * src{z});
            end
        end
    end

    S = {[b{2}, b{1}; b{4}, b{3}]; D; D; 1};
end

function q = outgoing_kz(val_sq)
    q = sqrt(val_sq + 0i);
    flip_mask = (imag(q) < -1e-14) | ((real(q) < 0) & (abs(imag(q)) < 1e-14));
    q(flip_mask) = -q(flip_mask);
end
