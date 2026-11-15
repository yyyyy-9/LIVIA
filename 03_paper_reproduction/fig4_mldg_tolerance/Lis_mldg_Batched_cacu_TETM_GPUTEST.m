% GPU cascade and TE/TM thickness-tolerance sweep for Fig. 4.

clc; 
clear; 
close all;

run(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'setup_paths.m'));

D_period   = 0.9;
wavelength = 1.065;

num_orders = 100;
nn         = num_orders;

theta      = asind(wavelength/(2*D_period)); 
nAir  = 1.0;
n_sub = 1.51;

n_SiO2  = 1.46;            
n_Ta2O5 = 2.14;    

hg      = 1;
f       = 0.54;
L       = D_period*f;

k0          = nAir*sin(theta*pi/180);

output_dir = fullfile(fileparts(mfilename('fullpath')), 'outputs');
num_pts = 100;
if strcmp(getenv('LIVIA_QUICK_DEMO'), '1')
    num_pts = 5;
    output_dir = fullfile(output_dir, 'quick');
end
if ~isfolder(output_dir)
    mkdir(output_dir);
end
t_H4_vec     = linspace(0.05, 0.15, num_pts);
t_L4_vec     = linspace(0.20, 0.35, num_pts);
[T_H4, T_L4] = meshgrid(t_H4_vec, t_L4_vec);
batch_size   = num_pts * num_pts;

h_H     = 0.1295;
h_L     = 0.1995;

htop    = 1.294;

hc       = htop - hg;

static_layers_top = [hc, 0, 0, 0.1353, 0.2000, 0.1878, 0.2226, 0.1212, 0.3479, h_H];
static_layers_bot = repmat([h_L, h_H], 1, 14);
d_base = [static_layers_top, static_layers_bot];

d_matrix = repmat(d_base, batch_size, 1);
d_matrix(:, 2) = T_H4(:);
d_matrix(:, 3) = T_L4(:);

n_stack = [n_SiO2, n_Ta2O5, n_SiO2, n_Ta2O5, n_SiO2, n_Ta2O5, n_SiO2, n_Ta2O5, n_SiO2, n_Ta2O5, repmat([n_SiO2, n_Ta2O5], 1, 14)];
% VPI: match grating exit and both film ports to substrate.
n_list  = [n_sub, n_stack, n_sub];

textures_g{1} = nAir;
textures_g{2} = n_sub; 
textures_g{3} = {[-L/2, L/2], [nAir, n_SiO2]};
profil_g      = {[1, hg, 0], [1, 3, 2]}; % Air -> Grating -> VPI (n_sub)

parm = res0(1); 
[aa_te_base, ~] = res1(wavelength, D_period, textures_g, nn, k0, parm);
S_GE_RCWA       = res2_reyy(aa_te_base, profil_g);
SgE_mat = S_GE_RCWA{1};

parm = res0(-1); 
[aa_tm_base, ~] = res1(wavelength, D_period, textures_g, nn, k0, parm);
S_GM_RCWA       = res2_reyy(aa_tm_base, profil_g);
SgM_mat = S_GM_RCWA{1};

if ~iscell(profil_g{1});
    profil_g={profil_g};
end;

kx_mo      = 2*pi/wavelength;
kx_vector  = kx_mo * sin(theta * pi/180);

ib    = profil_g{end}{2}(end); 
P_mat = aa_te_base{1}{ib}{1};

[dim_fourier, dim_modes] = size(P_mat); 
center_row = (dim_fourier - 1)/2 + 1;
for i = 1:dim_modes
    [~, max_idx] = max(abs(P_mat(:, i)));
    m = max_idx - center_row;
    modes_order_list(i) = m;
end
K_grating  = 2*pi / D_period;
kx_aligned = kx_vector + modes_order_list * K_grating;

N_modes = length(modes_order_list);
% Map RETICOLO blocks to Redheffer order.
% Grating [S11, S12, S21, S22] comes from RETICOLO [12, 11, 22, 21].
ESg11 = SgE_mat(1:N_modes, N_modes+1:2*N_modes);
ESg21 = SgE_mat(N_modes+1:2*N_modes, N_modes+1:2*N_modes);
ESg12 = SgE_mat(1:N_modes, 1:N_modes);
ESg22 = SgE_mat(N_modes+1:2*N_modes, 1:N_modes);

N_modes = length(modes_order_list);
MSg11 = SgM_mat(1:N_modes, N_modes+1:2*N_modes);
MSg21 = SgM_mat(N_modes+1:2*N_modes, N_modes+1:2*N_modes);
MSg12 = SgM_mat(1:N_modes, 1:N_modes);
MSg22 = SgM_mat(N_modes+1:2*N_modes, 1:N_modes);
[ESf11, ESf12, ESf21, ESf22] = FastBatch_SMatrix_vd(kx_aligned, wavelength, d_matrix, n_list, 'TE');
[MSf11, MSf12, MSf21, MSf22] = FastBatch_SMatrix_vd(kx_aligned, wavelength, d_matrix, n_list, 'TM');

tic;
% Limit device memory by transferring and cascading one film batch at a time.
gpu_chunk_size = 1000; 

num_batches = size(ESf11, 1);
N = size(ESg11, 1);

% Retain all four output blocks in host memory.
ES11_tot = zeros(N, N, num_batches);
ES12_tot = zeros(N, N, num_batches);
ES21_tot = zeros(N, N, num_batches);
ES22_tot = zeros(N, N, num_batches);

ESg11_gpu = gpuArray(ESg11);
ESg12_gpu = gpuArray(ESg12);
ESg21_gpu = gpuArray(ESg21);
ESg22_gpu = gpuArray(ESg22);

for c = 1:ceil(num_batches / gpu_chunk_size)
    idx_start = (c-1) * gpu_chunk_size + 1;
    idx_end = min(c * gpu_chunk_size, num_batches);
    
    ESf11_chunk = gpuArray(ESf11(idx_start:idx_end, :));
    ESf12_chunk = gpuArray(ESf12(idx_start:idx_end, :));
    ESf21_chunk = gpuArray(ESf21(idx_start:idx_end, :));
    ESf22_chunk = gpuArray(ESf22(idx_start:idx_end, :));
    
    [S11_c, S12_c, S21_c, S22_c] = Batched_Cascade_Grating_Film(...
        ESg11_gpu, ESg12_gpu, ESg21_gpu, ESg22_gpu, ...
        ESf11_chunk, ESf12_chunk, ESf21_chunk, ESf22_chunk);
    
    ES11_tot(:, :, idx_start:idx_end) = gather(S11_c);
    ES12_tot(:, :, idx_start:idx_end) = gather(S12_c);
    ES21_tot(:, :, idx_start:idx_end) = gather(S21_c);
    ES22_tot(:, :, idx_start:idx_end) = gather(S22_c);
    
    wait(gpuDevice);
end

clearvars ESg11_gpu ESg12_gpu ESg21_gpu ESg22_gpu ...
          ESf11_chunk ESf12_chunk ESf21_chunk ESf22_chunk ...
          S11_c S12_c S21_c S22_c;

MS11_tot = zeros(N, N, num_batches);
MS12_tot = zeros(N, N, num_batches);
MS21_tot = zeros(N, N, num_batches);
MS22_tot = zeros(N, N, num_batches);

MSg11_gpu = gpuArray(MSg11);
MSg12_gpu = gpuArray(MSg12);
MSg21_gpu = gpuArray(MSg21);
MSg22_gpu = gpuArray(MSg22);

for c = 1:ceil(num_batches / gpu_chunk_size)
    idx_start = (c-1) * gpu_chunk_size + 1;
    idx_end = min(c * gpu_chunk_size, num_batches);

    MSf11_chunk = gpuArray(MSf11(idx_start:idx_end, :));
    MSf12_chunk = gpuArray(MSf12(idx_start:idx_end, :));
    MSf21_chunk = gpuArray(MSf21(idx_start:idx_end, :));
    MSf22_chunk = gpuArray(MSf22(idx_start:idx_end, :));

    [S11_c, S12_c, S21_c, S22_c] = Batched_Cascade_Grating_Film(...
        MSg11_gpu, MSg12_gpu, MSg21_gpu, MSg22_gpu, ...
        MSf11_chunk, MSf12_chunk, MSf21_chunk, MSf22_chunk);

    MS11_tot(:, :, idx_start:idx_end) = gather(S11_c);
    MS12_tot(:, :, idx_start:idx_end) = gather(S12_c);
    MS21_tot(:, :, idx_start:idx_end) = gather(S21_c);
    MS22_tot(:, :, idx_start:idx_end) = gather(S22_c);

    wait(gpuDevice);
end

Batched_time = toc;

idx_minus_1 = find(modes_order_list == -1);
idx_0       = find(modes_order_list == 0);

RE_minus_1_complex = squeeze(ES11_tot(idx_minus_1, idx_0, :));
RE_minus_1_eff = abs(RE_minus_1_complex).^2; 

RM_minus_1_complex = squeeze(MS11_tot(idx_minus_1, idx_0, :));
RM_minus_1_eff = abs(RM_minus_1_complex).^2; 

clearvars ES11_tot ES12_tot ES21_tot ES22_tot ...
    MSf11_chunk MSf12_chunk MSf21_chunk MSf22_chunk ...

EffTE_2D = reshape(RE_minus_1_eff, [num_pts, num_pts]);
EffTM_2D = reshape(RM_minus_1_eff, [num_pts, num_pts]);

figure('Name', 'Tolerance Analysis', 'Color', 'w');
subplot(121)
contourf(T_H4, T_L4, EffTE_2D, 100, 'LineStyle', 'none');
colorbar;
colormap jet;
caxis([0.8 1.0]);
xlabel('Thickness of H4 (\mu m)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Thickness of L4 (\mu m)', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('Batched_Cascade (-1st TE)\nTime: %.2f s', Batched_time), 'FontSize', 14);

hold on;
contour(T_H4, T_L4, EffTE_2D, [0.97 0.97], 'LineColor', 'w', 'LineWidth', 2);

subplot(122)
contourf(T_H4, T_L4, EffTM_2D, 100, 'LineStyle', 'none');
colorbar;
colormap jet;
caxis([0.8 1.0]);
xlabel('Thickness of H4 (\mu m)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Thickness of L4 (\mu m)', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('Batched_Cascade (-1st TM)\nTime: %.2f s', Batched_time), 'FontSize', 14);

hold on;
contour(T_H4, T_L4, EffTM_2D, [0.97 0.97], 'LineColor', 'w', 'LineWidth', 2);

save(fullfile(output_dir, 'Batched_Cascade2.mat'), 'T_H4', 'T_L4', 'EffTE_2D', 'EffTM_2D', 'Batched_time');


