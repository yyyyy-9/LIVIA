% Compute zero-order TE transmission versus wavelength, slant, and offset for Fig. 5.
% Geometry and wavelengths are in micrometers. Angles are in degrees.

clc; 
clear; 
close all;

run(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'setup_paths.m'));
%% Scan grid
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
k0 = nSiO2 * sin(theta_inc * pi/180);

if strcmp(getenv('LIVIA_QUICK_DEMO'), '1')
    num_lambda = 5; num_theta = 7; num_dx = 51; % Quick test grid
else
    num_lambda = 81;  % Wavelength points (controls EVD count)
    num_theta  = 121; % Slant-angle points (controls slice cascade count)
    num_dx     = 1001;% Relative-offset points (batched Hadamard + star product)
end

lambda_array = linspace(1.60, 1.62, num_lambda);
theta_array  = linspace(45, 75, num_theta);
dx_array     = linspace(-D_period/2, D_period/2, num_dx);

% Array axes: slant angle, relative offset, wavelength.
T_3D_matrix = zeros(num_theta, num_dx, num_lambda);

%% Geometry and layer profiles
z_levels = ((1:num_layers)-0.5) * (hg/num_layers);
layerlist = (1:num_layers) + 3;
hlist = ones(1, num_layers) * hg / num_layers;
profil_in  = {[0, 0], [1, 3]};
profil_top = {[0, hlist, 0], [3, layerlist, 3]};
profil_cav = {[0, d_cavity, 0], [3, 3, 3]};
profil_bot = {[0, hlist, 0], [3, layerlist, 3]}; 
profil_out = {[0, 0], [3, 2]};

parm = res0(1); parm.not_io = 1; 

%% Transmission sweep
fprintf('======================================================\n');
fprintf('Transmission scan: %d wavelengths x %d slants x %d offsets.\n', num_lambda, num_theta, num_dx);
total_loops = num_lambda * num_theta;
current_loop = 0;
tic;

% Solve the reference texture modes once per wavelength.
for l_idx = 1:num_lambda
    current_lambda = lambda_array(l_idx);
    
    textures = {nSiO2, nSiO2, nLiq, {[-f*D_period/2, f*D_period/2], [nLiq, nSi]}};
    [aa, ~] = res1(current_lambda, D_period, textures, nn, k0, parm);
    alpha = aa{2}; if iscell(alpha), alpha = alpha{2}; end
    UN = aa{4};
    seed_te = aa{1}{4}; seed_w = seed_te{8}; 
    
    S_in  = res2_reyy(aa, profil_in);
    S_cav = res2_reyy(aa, profil_cav);
    S_out = res2_reyy(aa, profil_out);
    
    % Shift the slice modes and assemble both gratings once per slant.
    for t_idx = 1:num_theta
        current_theta = theta_array(t_idx);
        current_loop = current_loop + 1;
        
        shifts_top = z_levels / tand(current_theta);
        shifts_top = shifts_top - shifts_top(1); 
        shifts_bot_rev = -z_levels / tand(current_theta); 
        shifts_bot_rev = shifts_bot_rev - shifts_bot_rev(end); 
        
        aa_top = aa; aa_bot = aa;
        for i = 1:num_layers
            w_new_top = shift_waa_vectorized(seed_w, shifts_top(i), UN);
            aa_top{1}{i+3} = apply_phase_shift(seed_te, shifts_top(i), alpha, UN, w_new_top);
            
            w_new_bot = shift_waa_vectorized(seed_w, shifts_bot_rev(i), UN);
            aa_bot{1}{i+3} = apply_phase_shift(seed_te, shifts_bot_rev(i), alpha, UN, w_new_bot);
        end
        
        S_top = res2_reyy(aa_top, profil_top);
        S_bot_0 = res2_reyy(aa_bot, profil_bot);
        
        S_TOP_TOTAL = retss(S_in, retss(S_top, S_cav));
        S_BOT_TOTAL = retss(S_bot_0, S_out);
        
        % Translate the lower grating S matrix for all offsets and batch the final cascades.
        t_complex_vec = get_batched_t_complex(S_TOP_TOTAL, S_BOT_TOTAL, alpha, UN, dx_array);
        
        T_3D_matrix(t_idx, :, l_idx) = abs(t_complex_vec).^2;
    end
    
    elapsed_time = toc;
    estimated_total = elapsed_time / (l_idx / num_lambda);
    fprintf('Wavelength %d/%d (%5.1f%%), elapsed %5.1f s, remaining %5.1f s.\n', ...
        l_idx, num_lambda, (l_idx/num_lambda)*100, elapsed_time, estimated_total - elapsed_time);
end
timeall = toc;
fprintf('Transmission scan completed in %.2f s.\n', timeall);

output_dir = fullfile(fileparts(mfilename('fullpath')), 'outputs');
if strcmp(getenv('LIVIA_QUICK_DEMO'), '1')
    output_dir = fullfile(output_dir, 'quick');
end
if ~isfolder(output_dir)
    mkdir(output_dir);
end
output_file = fullfile(output_dir, 'VolumeData_3D_RoughScan.mat');
save(output_file, 'T_3D_matrix', 'lambda_array', 'theta_array', 'dx_array', 'timeall');
fprintf('Saved %s\n', output_file);

function t_complex_vec = get_batched_t_complex(S_A, S_B0, alpha, UN, dx_array)
    N_size = size(S_A{1}, 1) / 2;
    idx_1 = 1:N_size; idx_2 = (N_size+1):(2*N_size);

    R_bot_A  = S_A{1}(idx_2, idx_1); 
    T_down_A = S_A{1}(idx_2, idx_2); 

    R_top_B0  = S_B0{1}(idx_1, idx_2); 
    T_down_B0 = S_B0{1}(idx_2, idx_2); 

    num_dx = length(dx_array);
    alpha_vec = alpha(1:N_size); 
    d_alpha = alpha_vec(:) - alpha_vec(:).'; 
    dx_norm_3D = reshape(dx_array / UN, 1, 1, num_dx); 
    Phase_Mask_3D = exp(-1i * d_alpha .* dx_norm_3D); 

    R_top_B  = R_top_B0 .* Phase_Mask_3D;
    T_down_B = T_down_B0 .* Phase_Mask_3D;

    I_3D = eye(N_size); 
    Z = pageinv(I_3D - pagemtimes(R_bot_A, R_top_B)); 
    T_down_tot = pagemtimes(T_down_B, pagemtimes(Z, T_down_A));

    mid_idx = ceil(N_size/2);
    t_complex_vec = squeeze(T_down_tot(mid_idx, mid_idx, :)).'; 
end

function new_core = apply_phase_shift(core, dx, alpha, UN, w_new)
    new_core = core;
    phase = exp(-1i * alpha(:) * (dx/UN));
    P = spdiags(phase, 0, length(phase), length(phase));
    P_inv = spdiags(conj(phase), 0, length(phase), length(phase));
    new_core{1} = P * core{1};
    new_core{2} = core{2} * P_inv;
    new_core{3} = P * core{3};
    if length(core)>=4 && ~isempty(core{4}), new_core{4} = core{4} * P_inv; end
    if length(core)>=7, new_core{6} = P * core{6}; new_core{7} = P * core{7}; end
    new_core{8} = w_new; 
end

function w_new = shift_waa_vectorized(w_seed, dx, UN)
    old_coords = w_seed{1}; 
    old_mats   = w_seed{2};
    period     = old_coords(end); 
    dx_norm = dx / UN;
    walls_new = sort(mod(old_coords(1:end-1) + dx_norm, period));
    new_coords = [walls_new, period]; 
    x_starts = [0, new_coords(1:end-1)];
    x_mids   = (x_starts + new_coords) / 2;
    x_query = mod(x_mids - dx_norm, period);
    indices = sum(x_query(:) > old_coords(:)', 2) + 1;
    new_mats = old_mats(:, indices);
    w_new = {new_coords, new_mats};
end