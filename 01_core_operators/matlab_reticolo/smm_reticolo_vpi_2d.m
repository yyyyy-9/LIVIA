function S = smm_reticolo_vpi_2d(port, wavelength, d, n_list)
% Build a 2D RETICOLO film S-matrix on equal virtual-port media (n_top == n_bot).
% port is created by reticolo_orders('port', aa, tex_id, sens) without symmetry folding.
% Output follows RETICOLO block order: [T_up, R_top; R_bot, T_down].

    assert(abs(n_list(1) - n_list(end)) < 1e-12, ...
        'smm_reticolo_vpi_2d requires identical terminal media (n_VPI).');

    rho = sqrt(sum(port.transverse_k.^2, 2)).';
    te = cell(1, 4);
    tm = cell(1, 4);
    [te{:}] = FastBatch_SMatrix_vd(rho, wavelength, d, n_list, 'TE');
    [tm{:}] = FastBatch_SMatrix_vd(rho, wavelength, d, n_list, 'TM');

    blocks = repmat({complex(zeros(port.D))}, 1, 4);
    for i = 1:port.M
        k_vec = port.transverse_k(i, :).';
        if norm(k_vec) > 1e-12
            er = k_vec / norm(k_vec);
        else
            er = [1; 0];
        end
        es = [-er(2); er(1)];
        U = [es, er];

        ids = port.mode_ids(i, :);
        Q = port.down(1:2, :, i);

        for b = 1:4
            a_te = te{b}(i);
            a_tm = tm{b}(i);
            if b == 1 || b == 4
                % At equal port media, converting TM H_p to tangential E flips reflections
                a_tm = -a_tm;
            end
            B_xy = U * diag([a_te, a_tm]) * U.';
            blocks{b}(ids, ids) = Q \ (B_xy * Q);
        end
    end

    S = {[blocks{2}, blocks{1}; blocks{4}, blocks{3}]; port.D; port.D; 1};
end
