% Draw Fig. 3 from archived results or from locally generated sweeps.

clc; 
clear; 
close all;

run(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'setup_paths.m'));
%%
base = fileparts(mfilename('fullpath'));
outputs_dir = fullfile(base, 'outputs');
if ~isfolder(outputs_dir)
    mkdir(outputs_dir);
end
data_dir = fullfile(base, 'reference');
if strcmp(getenv('LIVIA_PLOT_GENERATED'), '1')
    data_dir = outputs_dir;
end

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


R_TE = abs(ESf11_raw).^2;
t_TE = abs(ESf21_raw);


fig = figure('Color', 'w', 'Position', [100, 100, 1400, 500]);

color_m2 = [0.8500 0.3250 0.0980];
color_p1 = [0.4940 0.1840 0.5560];

t_main = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

%% Panel a: Transverse-wavevector response
ax_a = nexttile(t_main, 1); 

plot(ax_a, kx_norm, t_TE, 'LineWidth', 1.5, 'Color', [0 0.4470 0.7410])
xlabel(ax_a, 'kx/k0', 'FontSize', 12, 'FontWeight', 'bold')
ylabel(ax_a, 'OTF', 'FontSize', 12, 'FontWeight', 'bold')
xlim(ax_a, [-3, 3]);
hold(ax_a, 'on');

x_range = xlim(ax_a); y_range = ylim(ax_a);
dx = x_range(2) - x_range(1); dy = y_range(2) - y_range(1);

arrow_length = 0.18 * dy;   
head_length  = 0.04 * dy;   
head_width   = 0.02 * dx;   

for i = 1:length(k_G)
    x_pos = k_G(i);
    y_tail = y_range(1);
    y_head = y_tail + arrow_length;
    
    plot(ax_a, [x_pos, x_pos], [y_tail, y_head - head_length], 'k-', 'LineWidth', 1.2);
    px = [x_pos, x_pos - head_width/2, x_pos + head_width/2];
    py = [y_head, y_head - head_length, y_head - head_length];
    patch(ax_a, px, py, 'k', 'EdgeColor', 'none');
    
    text(ax_a, x_pos, y_head + 0.02*dy, sprintf('n=%d', m_order(i)), ...
        'Color', 'k', 'FontSize', 11, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end
hold(ax_a, 'off');

text(ax_a, -0.22, 1.02, '{\bf a}', 'Units', 'normalized', 'FontSize', 22, 'FontName', 'Arial', 'VerticalAlignment', 'bottom');
title(ax_a,'')
%% Panel b: Wavelength scan
t_b = tiledlayout(t_main, 2, 1, 'TileSpacing', 'none', 'Padding', 'none');
t_b.Layout.Tile = 2;

ax_b1 = nexttile(t_b); 
load(fullfile(data_dir, 'f065_re1000.mat'), 'wvllist', 'te_f1', 'te_0');
plot(ax_b1, wvllist, te_f1, 'r-', 'LineWidth', 1.5); hold(ax_b1, 'on');
plot(ax_b1, wvllist, te_0, 'b--', 'LineWidth', 1.5);
plot(ax_b1, wvllist, te_f1+te_0, 'k-', 'LineWidth', 1.5); hold(ax_b1, 'off');
ylabel(ax_b1, 'Efficiency', 'FontSize', 14, 'FontWeight', 'bold');
ylim(ax_b1, [0 1.1]);
xlim(ax_b1, [min(wvllist) max(wvllist)]);
ax_b1.XTickLabel = [];
grid(ax_b1, 'on');

text(ax_b1, -0.17, 1.02, '{\bf b}', 'Units', 'normalized', 'FontSize', 22, 'FontName', 'Arial', 'VerticalAlignment', 'bottom');


ax_b2 = nexttile(t_b); 
load(fullfile(data_dir, 'layerwvl.mat'), 'wvllist', 'OTF_TE_m2');
plot(ax_b2, wvllist, log10(OTF_TE_m2(:,1)), '-', 'Color', color_m2, 'LineWidth', 1.5); hold(ax_b2, 'on');
plot(ax_b2, wvllist, log10(OTF_TE_m2(:,2)), '-', 'Color', color_p1, 'LineWidth', 1.5); hold(ax_b2, 'off');
xlabel(ax_b2, 'Wavelength \lambda (\mum)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel(ax_b2, 'Log_{10}(OTF)', 'FontSize', 14, 'FontWeight', 'bold');
xlim(ax_b2, [min(wvllist) max(wvllist)]);
grid(ax_b2, 'on');

danger_x = 1.07; 
xline(ax_b1, danger_x, 'k:', 'LineWidth', 1.5);
xline(ax_b2, danger_x, 'k:', 'LineWidth', 1.5); 



linkaxes([ax_b1, ax_b2], 'x');


%% Panel c: Thickness scaling
t_c = tiledlayout(t_main, 2, 1, 'TileSpacing', 'none', 'Padding', 'none');
t_c.Layout.Tile = 3;

ax_c1 = nexttile(t_c); 
load(fullfile(data_dir, 'f065_relayer.mat'), 'faclist', 'te_f1', 'te_0');
plot(ax_c1, faclist, te_f1, 'r-', 'LineWidth', 1.5); hold(ax_c1, 'on');
plot(ax_c1, faclist, te_0, 'b--', 'LineWidth', 1.5);
plot(ax_c1, faclist, te_f1+te_0, 'k-', 'LineWidth', 1.5); hold(ax_c1, 'off');
ylabel(ax_c1, 'Efficiency', 'FontSize', 14, 'FontWeight', 'bold');
ylim(ax_c1, [0 1.1]);
xlim(ax_c1, [min(faclist) max(faclist)]);
ax_c1.XTickLabel = []; 
grid(ax_c1, 'on');

text(ax_c1, -0.17, 1.02, '{\bf c}', 'Units', 'normalized', 'FontSize', 22, 'FontName', 'Arial', 'VerticalAlignment', 'bottom');

ax_c2 = nexttile(t_c); 
load(fullfile(data_dir, 'layerres.mat'), 'scale_factors', 'OTF_TE_m2');
plot(ax_c2, scale_factors, log10(OTF_TE_m2), '-', 'Color', color_m2, 'LineWidth', 2);
xlabel(ax_c2, 'Thickness Scaling (S)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel(ax_c2, 'Log_{10}(OTF)', 'FontSize', 14, 'FontWeight', 'bold');
xlim(ax_c2, [min(scale_factors) max(scale_factors)]);
grid(ax_c2, 'on');

linkaxes([ax_c1, ax_c2], 'x');


%% Remove overlapping tick labels at the panel boundaries

legend(ax_c1, 'n = -1', 'n = 0', 'Total', 'Location', 'best', 'FontSize', 10);

legend(ax_b2, 'n = -2', 'n = +1', 'Location', 'best', 'FontSize', 10);

set([ax_a, ax_b1, ax_b2, ax_c1, ax_c2], 'FontName', 'Arial', 'FontSize', 13, 'LineWidth', 1.2, 'Box', 'on');

drawnow;

labels_b = yticklabels(ax_b2);
if ~isempty(labels_b)
    labels_b{end} = '';
    yticklabels(ax_b2, labels_b);
end

labels_c = yticklabels(ax_c2);
if ~isempty(labels_c)
    labels_c{end} = '';
    yticklabels(ax_c2, labels_c);
end

title(t_main, ' ', 'FontSize', 24);

exportgraphics(fig, fullfile(outputs_dir, 'Figure4_Composite_Raw.pdf'), 'ContentType', 'vector');
