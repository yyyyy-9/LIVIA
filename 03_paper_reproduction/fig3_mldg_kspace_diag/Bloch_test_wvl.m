% Film transmission amplitude at the m = -2 and +1 wavevectors for Fig. 3b.

clc; 
clear; 
close all;

run(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'setup_paths.m'));
%%

D_period   = 0.9;
wavelength = 1.065;


theta      = asind(1.065/(2*D_period)); 

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



num_pairs     = 20;

huplayer      = [hc, 0.1003, 0.2687, 0.1353, 0.2000, 0.1878, 0.2226, 0.1212, 0.3479, h_H]*0;


d_list        = [repmat([h_L, h_H], 1, num_pairs)];
n_stack_inner = [nAir, repmat([n_SiO2, n_Ta2O5], 1, num_pairs), n_sub];
%%

%% Wavelength scan at the m = -2 and +1 wavevectors

m_target = [-2, +1];

wvllist = (1000:0.01:1100) * 1e-3;

fprintf('Evaluating film transmission at %d wavelengths.\n', length(wvllist));

OTF_TE_m2 = zeros(length(wvllist), length(m_target)); 

tic;
parfor wvi = 1:length(wvllist)
    wl = wvllist(wvi);

    k0        = 2*pi/wl;
    k_G       = (k0*sind(theta) + (m_target)*2*pi/(D_period))/k0;
    kx_m2     = k_G * k0;

    [~, ~, ESf21_batch, ~] = FastBatch_SMatrix_vd(kx_m2, wl, d_list, n_stack_inner, 'TE');

    OTF_TE_m2(wvi,:) = abs(ESf21_batch); 
end
toc;

%% Plot the film transmission amplitudes

figure('Color','w','Position',[300, 200, 800, 500]);

plot(wvllist*1000, log10(OTF_TE_m2(:,1)), 'r-', 'LineWidth', 2); hold on;
plot(wvllist*1000, log10(OTF_TE_m2(:,2)), 'b-', 'LineWidth', 2); hold off;

output_dir = fullfile(fileparts(mfilename('fullpath')), 'outputs');
if ~isfolder(output_dir)
    mkdir(output_dir);
end
save(fullfile(output_dir, 'layerwvl.mat'), 'wvllist', 'OTF_TE_m2');

xlabel('Wavelength \lambda (nm)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Log_{10}(OTF)', 'FontSize', 14, 'FontWeight', 'bold');
title('Littrow Symmetry Breaking: Evanescent Probe Dispersion', 'FontSize', 16);

legend('m = -2 probe', 'm = +1 probe', 'Location', 'best', 'FontSize', 12);

grid on;
axis tight;
set(gca, 'FontName', 'Arial', 'FontSize', 14, 'LineWidth', 1.2);
