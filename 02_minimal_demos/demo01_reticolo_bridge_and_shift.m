% Verification of LIVIA <-> RETICOLO bridges and lateral translation operators:
%   1. 1D VPI: MLDG + Auto_Scan_S_Matrix_v2 & interleaved G1 * S_film * G2
%   2. 2D Conical VPI: smm_reticolo_vpi_2d + reticolo_orders
%   3. Explicit native-retb port bridge (reticolo_film_smm) on unequal ports (1D & 2D)
%   4. 1D slice-level (shift_reticolo_layer_1d) & subsystem-level (shift_reticolo_smatrix_1d) shifts

clear; clc;
repo_root = fileparts(fileparts(mfilename('fullpath')));
run(fullfile(repo_root, 'setup_paths.m'));

if exist('res0', 'file') ~= 2
    error('RETICOLO is not on the MATLAB path. Set RETICOLO_ROOT before running demo01.');
end

fprintf('RETICOLO bridge & translation verification:\n');

%% 1. 1D VPI: MLDG & Interleaved Grating-Film-Grating
D_period   = 0.9;
wavelength = 1.065;
nn         = 40;
theta      = asind(wavelength / (2 * D_period));
nAir = 1.0; n_SiO2 = 1.46; n_Ta2O5 = 2.14; n_sub = 1.51;
hg = 1.0; f = 0.54; L = D_period * f;
k0_inc = nAir * sind(theta);
k0_vac = 2 * pi / wavelength;

hc = 1.294 - hg;
d_film = [hc, 0.1003, 0.2687, 0.1353, 0.2000, 0.1878, 0.2226, 0.1212, 0.3479, 0.1295, ...
          repmat([0.1995, 0.1295], 1, 14)];
n_list = [n_sub, repmat([n_SiO2, n_Ta2O5], 1, 19), n_sub];

textures_1d = {nAir, n_SiO2, n_Ta2O5, n_sub, ...
               {[-L/2, L/2], [nAir, n_SiO2]}, {[-0.24, 0.24], [n_SiO2, n_Ta2O5]}};
profil_mldg = {[0, hg, d_film, 0], [1, 5, repmat([2, 3], 1, 19), 4]};
profil_g    = {[0, hg, 0], [1, 5, 4]};

d_int = repmat([0.18, 0.12], 1, 3);
n_int = [n_SiO2, repmat([n_Ta2O5, n_SiO2], 1, 3), n_SiO2];
profil_int_full = {[0, 0.35, d_int, 0.40, 0], [1, 5, repmat([3, 2], 1, 3), 6, 4]};

for pol_val = [1, -1]
    if pol_val == 1
        pol_name = 'TE';
    else
        pol_name = 'TM';
    end
    parm = res0(pol_val);
    parm.not_io = 1;

    aa = res1(wavelength, D_period, textures_1d, nn, k0_inc, parm);
    ef_ref = res2(aa, profil_mldg, parm);
    S_ref  = res2_reyy(aa, profil_mldg, parm);
    eta_ref = ef_ref.inc_top_reflected.efficiency(ef_ref.inc_top_reflected.order == -1);

    S_g_cell = res2_reyy(aa, profil_g, parm);
    Sg_mat = S_g_cell{1};
    orders_sub = port_orders_1d(aa, 4);
    kx_sub = k0_vac * sind(theta) + orders_sub * (2 * pi / D_period);
    dim_m = numel(orders_sub);

    Sg11 = Sg_mat(1:dim_m, dim_m+1:2*dim_m);
    Sg12 = Sg_mat(1:dim_m, 1:dim_m);
    Sg21 = Sg_mat(dim_m+1:2*dim_m, dim_m+1:2*dim_m);
    Sg22 = Sg_mat(dim_m+1:2*dim_m, 1:dim_m);

    [Sf11, Sf12, Sf21, Sf22] = FastBatch_SMatrix_vd(kx_sub, wavelength, d_film, n_list, pol_name);
    [S11_t, S12_t, S21_t, S22_t] = Batched_Cascade_Grating_Film( ...
        Sg11, Sg12, Sg21, Sg22, Sf11, Sf12, Sf21, Sf22);
    S_tot_ret = [S12_t, S11_t; S22_t, S21_t];

    scan_res = Auto_Scan_S_Matrix_v2(S_tot_ret, aa, {profil_mldg});
    eta_livia = scan_res.R_up(scan_res.R_up(:, 1) == -1, 2);

    G1 = res2_reyy(aa, {[0, 0.35, 0], [1, 5, 2]}, parm);
    G2 = res2_reyy(aa, {[0, 0.40, 0], [2, 6, 4]}, parm);
    kx_sio2 = k0_vac * sind(theta) + port_orders_1d(aa, 2) * (2 * pi / D_period);
    [r, tu, td, rb] = FastBatch_SMatrix_vd(kx_sio2, wavelength, d_int, n_int, pol_name);
    S_film_int = {[diag(tu), diag(r); diag(rb), diag(td)]; dim_m; dim_m; 1};
    S_int_hyb  = retss(retss(G1, S_film_int), G2);
    S_int_ref  = res2_reyy(aa, profil_int_full, parm);

    err_eta    = abs(eta_livia - eta_ref);
    err_mldg_s = max(abs(S_tot_ret - S_ref{1}), [], 'all');
    err_int_s  = max(abs(S_int_hyb{1} - S_int_ref{1}), [], 'all');
    fprintf('  1D VPI (%s): MLDG |d_eta| = %.2e, max|dS| = %.2e | Interleaved max|dS| = %.2e\n', ...
        pol_name, err_eta, err_mldg_s, err_int_s);
    assert(err_eta < 1e-11 && err_mldg_s < 1e-10 && err_int_s < 1e-11);
end

%% 2. 2D Conical VPI: smm_reticolo_vpi_2d + reticolo_orders
wl2 = 0.532; per2 = [2, 2]; N2 = 2; th2 = 30; ph2 = 30;
nI = 1.0; nII = 1.4607; nInc = 0.93878 + 6.4195i;
film_d2 = [0.01, 0.1, 0.1, 0.1];
film_n2 = [nII, nInc, nII, 2.6678, 1.9044, nII];

tex2 = {nI, nII, {nI, [0, 0, 1, 1, nInc, 1]}, nInc, 2.6678, 1.9044};
parm2 = res0;
parm2.not_io = 1;
parm2.res1.ftemp = 0;
parm2.res1.champ = 1;
parm2.sym.x = [];
parm2.sym.y = [];
aa2 = res1(wl2, per2, tex2, [N2, N2], nI * sind(th2), ph2, parm2);

top2 = reticolo_orders('port', aa2, 1, parm2.res2.tolh);
bot2 = reticolo_orders('port', aa2, 2, -parm2.res2.tolb);
S_g2    = res2_reyy(aa2, {[0, 0.6, 0], [1, 3, 2]}, parm2);
S_film2 = smm_reticolo_vpi_2d(bot2, wl2, film_d2, film_n2);
S_dev2  = retss(S_g2, S_film2);
S_nat2  = res2_reyy(aa2, {[0, 0.6, film_d2, 0], [1, 3, 4, 2, 5, 6, 2]}, parm2);
ef2     = res2(aa2, {[0, 0.6, film_d2, 0], [1, 3, 4, 2, 5, 6, 2]}, parm2);

src2  = reticolo_orders('source', top2, [0, 0], ph2);
D2    = top2.D;
R2    = S_dev2{1}(1:D2, D2+1:2*D2);
q_m10 = reticolo_orders('read', R2, top2, src2, [-1, 0], 'up');
k_te  = find(all(ef2.TEinc_top_reflected.order == [-1, 0], 2), 1);
k_tm  = find(all(ef2.TMinc_top_reflected.order == [-1, 0], 2), 1);
err_2d_s   = max(abs(S_dev2{1} - S_nat2{1}), [], 'all');
err_2d_eta = max(abs(q_m10.eta - [ef2.TEinc_top_reflected.efficiency(k_te), ef2.TMinc_top_reflected.efficiency(k_tm)]));
fprintf('  2D Conical VPI (N=%d, D=%d): max|dS| = %.2e, order [-1,0] |d_eta| = %.2e\n', ...
    N2, D2, err_2d_s, err_2d_eta);
assert(err_2d_s < 1e-10 && err_2d_eta < 1e-11);

%% 3. Explicit Native-retb Bridge on Unequal Ports (PMMA -> Ag)
wl3 = 0.365; per3 = 0.4; w3 = 0.20;
nGlass = sqrt(1.46); nPMMA = 1.5; nCr = sqrt(-8.55 + 8.96i); nAg = sqrt(-2.4 + 0.25i); nPR = 1.6;
d3 = [0.05, 0.02, 0.04];
n3 = [nPMMA, nPMMA, nAg, nPR, nAg];

err_bridge_1d = 0;
for pol_val = [1, -1]
    p3 = res0(pol_val);
    p3.not_io = 1;
    p3.res1.ftemp = 0;
    p3.res1.champ = 1;
    tex3_1d = {nGlass, nAg, {[-w3/2, w3/2], [nCr, nPMMA]}, nPMMA, nAg, nPR};
    aa3_1d  = res1(wl3, per3, tex3_1d, 8, 0, p3);
    B3_1d   = reticolo_film_smm(aa3_1d, 4, 5, wl3, d3, n3, p3);
    S3_hyb  = retss(res2_reyy(aa3_1d, {[0, 0.05, 0], [1, 3, 4]}, p3), B3_1d);
    S3_nat  = res2_reyy(aa3_1d, {[0, 0.05, d3, 0], [1, 3, 4, 5, 6, 5]}, p3);
    err_bridge_1d = max(err_bridge_1d, max(abs(S3_hyb{1} - S3_nat{1}), [], 'all'));
end

p3_2d = res0;
p3_2d.not_io = 1;
p3_2d.res1.ftemp = 0;
p3_2d.res1.champ = 1;
p3_2d.sym.x = [];
p3_2d.sym.y = [];
tex3_2d = {nGlass, nAg, {nCr, [0, 0, w3, w3, nPMMA, 1]}, nPMMA, nAg, nPR};
aa3_2d  = res1(wl3, [per3, per3], tex3_2d, [2, 2], 0, 0, p3_2d);
B3_2d   = reticolo_film_smm(aa3_2d, 4, 5, wl3, d3, n3, p3_2d);
S3_hyb2 = retss(res2_reyy(aa3_2d, {[0, 0.05, 0], [1, 3, 4]}, p3_2d), B3_2d);
S3_nat2 = res2_reyy(aa3_2d, {[0, 0.05, d3, 0], [1, 3, 4, 5, 6, 5]}, p3_2d);
err_bridge_2d = max(abs(S3_hyb2{1} - S3_nat2{1}), [], 'all');
fprintf('  Explicit retb bridge (PMMA -> Ag): 1D max|dS| = %.2e, 2D max|dS| = %.2e\n', ...
    err_bridge_1d, err_bridge_2d);
assert(err_bridge_1d < 1e-11 && err_bridge_2d < 1e-11);

%% 4. 1D Slice-Level & Subsystem-Level Translation Reuse
dx_shift = 0.21 * D_period;
parm_te = res0(1);
parm_te.not_io = 1;
tex_2g = {nAir, n_SiO2, {[-0.18, 0.18], [nAir, n_Ta2O5]}, {[-0.22, 0.22], [nAir, n_SiO2]}};
aa_base = res1(wavelength, D_period, tex_2g, 25, k0_inc, parm_te);

% 4a. Slice-level descriptor shift (shift_reticolo_layer_1d)
aa_layer_sft = aa_base;
aa_layer_sft{1}{3} = shift_reticolo_layer_1d(aa_base{1}{3}, dx_shift, aa_base{2}, aa_base{4});
S_layer_sft = res2_reyy(aa_layer_sft, {[0, 0.35, 0.08, 0.45, 0], [1, 3, 1, 4, 2]}, parm_te);

% 4b. Subsystem-level S-matrix shift (shift_reticolo_smatrix_1d)
S_upper = res2_reyy(aa_base, {[0, 0.35, 0.08], [1, 3, 1]}, parm_te);
S_lower = res2_reyy(aa_base, {[0, 0.45, 0],    [1, 4, 2]}, parm_te);
S_sub_sft = retss(shift_reticolo_smatrix_1d(S_upper, dx_shift, aa_base{2}, aa_base{4}), S_lower);

tex_2g_sft = {nAir, n_SiO2, {[-0.18, 0.18] + dx_shift, [nAir, n_Ta2O5]}, {[-0.22, 0.22], [nAir, n_SiO2]}};
aa_sft = res1(wavelength, D_period, tex_2g_sft, 25, k0_inc, parm_te);
S_rebuilt = res2_reyy(aa_sft, {[0, 0.35, 0.08, 0.45, 0], [1, 3, 1, 4, 2]}, parm_te);

err_layer_sft = max(abs(S_layer_sft{1} - S_rebuilt{1}), [], 'all');
err_sub_sft   = max(abs(S_sub_sft{1}   - S_rebuilt{1}), [], 'all');
fprintf('  1D shifts vs. rebuilt geometry: layer max|dS| = %.2e, subsystem max|dS| = %.2e\n', ...
    err_layer_sft, err_sub_sft);
assert(err_layer_sft < 1e-12 && err_sub_sft < 1e-12);

function orders_m = port_orders_1d(aa, tex_id)
    P_mat = aa{1}{tex_id}{1};
    [dim_f, ~] = size(P_mat);
    center_row = (dim_f - 1) / 2 + 1;
    [~, midx] = max(abs(P_mat), [], 1);
    orders_m = midx - center_row;
end
