% Compute the native RETICOLO spectrum at fixed slant and offset for Fig. 5b.
% Geometry and wavelengths are in micrometers. Angles are in degrees.

clc; 
clear; 
close all;

run(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'setup_paths.m'));
%% Fixed geometry and wavelength grid
D_period   = 0.9;   
theta_inc  = 0;     
num_orders = 50;    
nn         = num_orders;
num_layers = 100;    
hg         = 0.2; 
d_cavity   = 0.3; 
f          = 1/3; 
nSiO2 = 1.45;  
nSi   = 3.47;  
nLiq  = 1.331; 

theta_target = 60;
dx_target    = 0.0;

output_dir = fullfile(fileparts(mfilename('fullpath')), 'outputs');
num_wavelengths = 301;
if strcmp(getenv('LIVIA_QUICK_DEMO'), '1')
    num_wavelengths = 5;
    output_dir = fullfile(output_dir, 'quick');
end
if ~isfolder(output_dir)
    mkdir(output_dir);
end
lambda_array = linspace(1.60, 1.62, num_wavelengths);
T_down_array = zeros(length(lambda_array), 1);

%% Layer profiles
z_levels = ((1:num_layers)-0.5) * (hg/num_layers);

shifts_top = z_levels / tand(theta_target);
shifts_top = shifts_top - shifts_top(1); 

shifts_bot_rev = -z_levels / tand(theta_target); 
shifts_bot_rev = shifts_bot_rev - shifts_bot_rev(end) + dx_target; 

textures_full = cell(1, 2*num_layers + 3);
textures_full{1} = nSiO2; textures_full{2} = nSiO2; textures_full{3} = nLiq;

for i = 1:num_layers
    lx_left  = -f*D_period/2 + shifts_top(i);
    textures_full{i+3} = {[lx_left, lx_left + f*D_period], [nLiq, nSi]};
end
for i = 1:num_layers
    lx_left  = -f*D_period/2 + shifts_bot_rev(i);
    lx_left = mod(lx_left + D_period/2, D_period) - D_period/2;
    textures_full{num_layers + i + 3} = {[lx_left, lx_left + f*D_period], [nLiq, nSi]};
end

layerlist_top = (1:num_layers) + 3;
layerlist_bot = (1:num_layers) + num_layers + 3;
hlist = ones(1, num_layers) * hg / num_layers;
profil_full = {[0, 0, hlist, d_cavity, hlist, 0, 0], [1, 3, layerlist_top, 3, layerlist_bot, 3, 2]};
parm = res0(1); parm.not_io = 1; 

%% Native RETICOLO wavelength sweep
fprintf('======================================================\n');
fprintf('RETICOLO spectrum at slant %.1f deg and offset %.2f um.\n', theta_target, dx_target);
tic;

for step = 1:length(lambda_array)
    current_lambda = lambda_array(step);
    k0 = nSiO2 * sin(theta_inc * pi/180); 
    
    [aa_full, ~] = res1(current_lambda, D_period, textures_full, nn, k0, parm);
    [ef_full, ~] = res2(aa_full, profil_full);
    T_down_array(step) = ef_full.inc_top_transmitted.efficiency{0};
    disp(current_lambda)
end
twavesweep = toc;
save(fullfile(output_dir, 'wavexisao_nopar.mat'), 'T_down_array', 'lambda_array', 'twavesweep');
return
%% Plot the spectrum
figure('Color','w','Position',[200, 200, 600, 400]);
plot(lambda_array, T_down_array, 'r-', 'LineWidth', 2);
xlabel('Wavelength (\mu m)', 'FontSize', 12);
ylabel('0-th Order Transmission', 'FontSize', 12);
title(sprintf('BIC Verification: \\theta = %.1f^\\circ, \\Delta x = %.2f \\mu m', theta_target, dx_target), 'FontSize', 14);
grid on;

