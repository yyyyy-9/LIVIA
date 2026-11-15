% Film-only transverse-wavevector response and thickness-scale sweep for Fig. 3.

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
k0        = 2*pi/wavelength;
kx_norm   = linspace(-3.0, 3.0, 20001);
kx_vector = kx_norm * k0 ;

m_order   = -3:2;
k_G       = (k0*sind(theta) + m_order*2*pi/(D_period))/k0;

%%

[ESf11_raw, ~, ESf21_raw, ~] = FastBatch_SMatrix_vd(kx_vector, wavelength, d_list, n_stack_inner, 'TE');
[MSf11_raw, ~, MSf21_raw, ~] = FastBatch_SMatrix_vd(kx_vector, wavelength, d_list, n_stack_inner, 'TM');


R_TE = abs(ESf11_raw).^2;
t_TE = abs(ESf21_raw);
R_TM = abs(MSf11_raw).^2;
t_TM = abs(MSf21_raw);



figure('Color','w','Position',[500, 100, 1400, 400]);

%% TE wavevector response
subplot(121)
plot(kx_norm, t_TE, 'LineWidth', 1.5, 'Color', [0 0.4470 0.7410])
xlabel('kx/k0', 'FontSize', 12, 'FontWeight', 'bold')
ylabel('OTF', 'FontSize', 12, 'FontWeight', 'bold')
title('TE Polarization', 'FontSize', 14)
xlim([-3, 3]);
hold on;

x_range = xlim;
y_range = ylim;
dx = x_range(2) - x_range(1);
dy = y_range(2) - y_range(1);

arrow_length = 0.20 * dy;
head_length  = 0.04 * dy;
head_width   = 0.02 * dx;

for i = 1:length(k_G)
    x_pos = k_G(i);
    
    y_tail = y_range(1);
    y_head = y_tail + arrow_length;
    
    plot([x_pos, x_pos], [y_tail, y_head - head_length], 'k-', 'LineWidth', 1.2);
    
    px = [x_pos, x_pos - head_width/2, x_pos + head_width/2];
    py = [y_head, y_head - head_length, y_head - head_length];
    patch(px, py, 'k', 'EdgeColor', 'none');
    
    text(x_pos, y_head + 0.02*dy, sprintf('m=%d', m_order(i)), ...
        'Color', 'k', 'FontSize', 11, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end
hold off;
text(-0.17, 1, '{\bf a}', 'Units', 'normalized', 'FontSize', 22, 'FontName', 'Arial', 'VerticalAlignment', 'bottom');


%% TM wavevector response
subplot(122)
plot(kx_norm, t_TM, 'LineWidth', 1.5, 'Color', [0.8500 0.3250 0.0980])
xlabel('kx/k0', 'FontSize', 12, 'FontWeight', 'bold')
ylabel('OTF', 'FontSize', 12, 'FontWeight', 'bold')
title('TM Polarization', 'FontSize', 14)
xlim([-3, 3]);
hold on;

x_range = xlim;
y_range = ylim;
dx = x_range(2) - x_range(1);
dy = y_range(2) - y_range(1);

arrow_length = 0.20 * dy;   
head_length  = 0.04 * dy;   
head_width   = 0.02 * dx;   

for i = 1:length(k_G)
    x_pos = k_G(i);
    
    y_tail = y_range(1);
    y_head = y_tail + arrow_length;
    
    plot([x_pos, x_pos], [y_tail, y_head - head_length], 'k-', 'LineWidth', 1.2);
    
    px = [x_pos, x_pos - head_width/2, x_pos + head_width/2];
    py = [y_head, y_head - head_length, y_head - head_length];
    patch(px, py, 'k', 'EdgeColor', 'none');
    
    text(x_pos, y_head + 0.02*dy, sprintf('m=%d', m_order(i)), ...
        'Color', 'k', 'FontSize', 11, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end
hold off;

text(-0.17, 1, '{\bf b}', 'Units', 'normalized', 'FontSize', 22, 'FontName', 'Arial', 'VerticalAlignment', 'bottom');


hFigList=get(0,'Children');
for figi=1:length( hFigList )
    hAxesList=get(hFigList(figi),'Children');
    set(hAxesList,'FontName','Arial','FontSize',14,'lineWidth',1.2)
end


%%
%% Film-thickness scaling at the m = -2 wavevector

m_target = -2;
idx_m = find(m_order == m_target);
kx_m2 = k_G(idx_m) * k0;

% Resolve the narrow resonance on the same dense grid as the paper.
scale_factors = linspace(0.98, 1.02, 500001)';
d_batch = scale_factors * d_list; 



fprintf('Evaluating %d film-thickness scale factors.\n', numel(scale_factors));
tic;
[~, ~, ESf21_batch, ~] = FastBatch_SMatrix_vd(kx_m2, wavelength, d_batch, n_stack_inner, 'TE');
toc;

OTF_TE_m2 = abs(ESf21_batch);

figure('Color','w','Position',[300, 200, 800, 500]);
plot(scale_factors, OTF_TE_m2, 'LineWidth', 2, 'Color', [0.6350 0.0780 0.1840]);
xlabel('Thickness Scaling Factor (S)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('OTF at m=-2', 'FontSize', 14, 'FontWeight', 'bold');
title('Resonance Sweeping across the m=-2 Wavevector', 'FontSize', 16);
grid on;
axis tight

hold on;
xline(0.987, 'k--', 'LineWidth', 1.5, 'Label', 'Danger: S = 0.987', ...
    'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'left', ...
    'FontSize', 12, 'FontWeight', 'bold');
hold off;

set(gca, 'FontName', 'Arial', 'FontSize', 14, 'LineWidth', 1.2);

output_dir = fullfile(fileparts(mfilename('fullpath')), 'outputs');
if ~isfolder(output_dir)
    mkdir(output_dir);
end
save(fullfile(output_dir, 'layerres.mat'), 'scale_factors', 'OTF_TE_m2');
