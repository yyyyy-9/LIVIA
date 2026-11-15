% Full-operator RETICOLO thickness-tolerance reference for Fig. 4.
% Assemble all four blocks for each design and save the target-order efficiency.

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

h_H     = 0.1295;
h_L     = 0.1995;

htop    = 1.294;

hc       = htop - hg;

output_dir = fullfile(fileparts(mfilename('fullpath')), 'outputs');
num_pts = 100;
if strcmp(getenv('LIVIA_QUICK_DEMO'), '1')
    num_pts = 5;
    output_dir = fullfile(output_dir, 'quick');
end
if ~isfolder(output_dir)
    mkdir(output_dir);
end
t_H4_vec = linspace(0.05, 0.15, num_pts); 
t_L4_vec = linspace(0.20, 0.35, num_pts); 
[T_H4, T_L4] = meshgrid(t_H4_vec, t_L4_vec);

H4_flat = T_H4(:);
L4_flat = T_L4(:);
batch_size = length(H4_flat);

eff_te_minus1 = zeros(batch_size, 1);
eff_tm_minus1 = zeros(batch_size, 1);

k0          = nAir*sin(theta*pi/180);

textures{1} = nAir;
textures{2} = n_sub;
textures{3} = {[-L/2, L/2], [nAir, n_SiO2]};

textures{4} = n_SiO2;
textures{5} = n_Ta2O5;

parm       = res0(1);
[aa_te, ~] = res1(wavelength, D_period, textures, nn, k0, parm);
parm       = res0(-1);
[aa_tm, ~] = res1(wavelength, D_period, textures, nn, k0, parm);

num_pairs     = 14;
d_list        = [repmat([h_L, h_H], 1, num_pairs)];
n_stack_inner = repmat([4, 5], 1, num_pairs+5);

delete(gcp("nocreate"));
parpool('Threads',16)

fprintf('Running RETICOLO parfor for %d designs.\n', batch_size);

tic

parfor i = 1:batch_size
    curr_H4 = H4_flat(i);
    curr_L4 = L4_flat(i);

    curr_huplayer = [hc, curr_H4, curr_L4, 0.1353, 0.2000, 0.1878, 0.2226, 0.1212, 0.3479, h_H];
    curr_d_list   = [curr_huplayer, d_list];

    curr_profil = {[1, hg, curr_d_list, 1], [1, 3, n_stack_inner, 2]};

    ef_te = res2_reyy(aa_te, curr_profil);
    ef_tm = res2_reyy(aa_tm, curr_profil);

    Res_SE_RCWA = Auto_Scan_S_Matrix_v2(ef_te{1}, aa_te, {curr_profil}).R_up(1,2);
    Res_SM_RCWA = Auto_Scan_S_Matrix_v2(ef_tm{1}, aa_tm, {curr_profil}).R_up(1,2);

    eff_te_minus1(i) = Res_SE_RCWA;
    eff_tm_minus1(i) = Res_SM_RCWA;

end
time_reticoloall = toc;
fprintf('RETICOLO parfor completed in %.4f s.\n', time_reticoloall);

Eff_TE2D_RETICOLO = reshape(eff_te_minus1, [num_pts, num_pts]);
Eff_TM2D_RETICOLO = reshape(eff_tm_minus1, [num_pts, num_pts]);

figure('Name', 'Traditional parfor RCWA Baseline', 'Color', 'w');
subplot(121)
contourf(T_H4, T_L4, Eff_TE2D_RETICOLO, 100, 'LineStyle', 'none');
colorbar; colormap jet; caxis([0.8 1.0]);
xlabel('Thickness of H4 (\mu m)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Thickness of L4 (\mu m)', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('RETICOLO parfor (-1st TE)\nTime: %.2f s', time_reticoloall), 'FontSize', 14);
hold on;
contour(T_H4, T_L4, Eff_TE2D_RETICOLO, [0.97 0.97], 'LineColor', 'w', 'LineWidth', 2);

subplot(122)
contourf(T_H4, T_L4, Eff_TM2D_RETICOLO, 100, 'LineStyle', 'none');
colorbar; colormap jet; caxis([0.8 1.0]);
xlabel('Thickness of H4 (\mu m)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Thickness of L4 (\mu m)', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('RETICOLO parfor (-1st TE)\nTime: %.2f s', time_reticoloall), 'FontSize', 14);
hold on;
contour(T_H4, T_L4, Eff_TM2D_RETICOLO, [0.97 0.97], 'LineColor', 'w', 'LineWidth', 2);

save(fullfile(output_dir, 'RETICOLOresallorder.mat'), 'T_H4', 'T_L4', 'Eff_TE2D_RETICOLO', 'Eff_TM2D_RETICOLO', 'time_reticoloall');
