% Figure 5. Draw the archived cross-sections or a locally generated volume.
clc; clear; close all;

base = fileparts(mfilename('fullpath'));
reference_dir = fullfile(base, 'reference');
output_dir = fullfile(base, 'outputs');
data_dir = reference_dir;
use_generated = strcmp(getenv('LIVIA_PLOT_GENERATED'), '1');
if use_generated
    if strcmp(getenv('LIVIA_QUICK_DEMO'), '1')
        output_dir = fullfile(output_dir, 'quick');
    end
    data_dir = output_dir;
end
if ~isfolder(output_dir)
    mkdir(output_dir);
end
%% Load reference and sweep data
set(0, 'DefaultAxesFontName', 'Arial'); 
set(0, 'DefaultTextFontName', 'Arial');

% The archived data stores only the five cuts used in the figure.
if use_generated
    data = load(fullfile(data_dir, 'VolumeData_3D_RoughScan.mat'), ...
        'T_3D_matrix', 'lambda_array', 'theta_array', 'dx_array');
    [~, data.theta_indices] = min(abs(data.theta_array(:) - [45, 60, 75]), [], 1);
    [~, data.dx_index] = min(abs(data.dx_array));
    [~, data.lambda_index] = min(abs(data.lambda_array - 1.61));
    data.T_fixed_theta = data.T_3D_matrix(data.theta_indices, :, :);
    data.T_fixed_dx = squeeze(data.T_3D_matrix(:, data.dx_index, :));
    data.T_fixed_lambda = data.T_3D_matrix(:, :, data.lambda_index);
    data = rmfield(data, 'T_3D_matrix');
else
    data = load(fullfile(reference_dir, 'figure5_slices.mat'));
end
lambda_array3D = data.lambda_array;
theta_array = data.theta_array;
dx_array = data.dx_array;

lambda_1d_full = linspace(1.600, 1.620, 301); 
COMSOL_full = [0.12947; 0.12918; 0.12889; 0.1286; 0.12829; 0.128; 0.12771; 0.12742; 0.12713; 0.12684; 0.12654; 0.12625; 0.12596; 0.12567; 0.12538; 0.12508; 0.12478; 0.12448; 0.12419; 0.12389; 0.12359; 0.1233; 0.123; 0.1227; 0.1224; 0.12197; 0.12167; 0.12137; 0.12106; 0.12076; 0.12045; 0.12014; 0.11983; 0.11953; 0.11922; 0.1189; 0.11859; 0.11827; 0.11795; 0.11764; 0.11732; 0.117; 0.11667; 0.11635; 0.11602; 0.11569; 0.11537; 0.11503; 0.1147; 0.11436; 0.11402; 0.11368; 0.11333; 0.11299; 0.11263; 0.11228; 0.11193; 0.11157; 0.11121; 0.11084; 0.11048; 0.11011; 0.10973; 0.10935; 0.10897; 0.10858; 0.10819; 0.10779; 0.10739; 0.10699; 0.10658; 0.10617; 0.10574; 0.10532; 0.10488; 0.10445; 0.104; 0.10355; 0.10309; 0.10262; 0.10215; 0.10167; 0.10118; 0.10069; 0.10018; 0.099662; 0.099136; 0.098601; 0.098053; 0.097494; 0.096923; 0.09633; 0.095733; 0.095122; 0.094496; 0.093857; 0.0932; 0.092526; 0.091833; 0.091122; 0.090387; 0.089634; 0.088859; 0.088059; 0.087235; 0.086384; 0.085504; 0.084593; 0.083651; 0.082674; 0.081659; 0.080605; 0.079508; 0.078366; 0.077176; 0.075933; 0.074634; 0.073274; 0.071849; 0.070353; 0.068782; 0.067126; 0.065379; 0.063533; 0.06158; 0.059511; 0.057303; 0.054963; 0.05247; 0.049819; 0.046974; 0.043928; 0.040665; 0.037172; 0.033422; 0.029411; 0.025137; 0.020633; 0.015922; 0.011138; 0.0065239; 0.002547; 1.65E-04; 0.0011991; 0.0094047; 0.033162; 0.088606; 0.20921; 0.44243; 0.77084; 0.98583; 0.95737; 0.81646; 0.67846; 0.57061; 0.49128; 0.43225; 0.38746; 0.35282; 0.32526; 0.30297; 0.28466; 0.26947; 0.25652; 0.24544; 0.23586; 0.22751; 0.22014; 0.2136; 0.20777; 0.20248; 0.19775; 0.19346; 0.18954; 0.18596; 0.18268; 0.17964; 0.17683; 0.17421; 0.17177; 0.1695; 0.16737; 0.16537; 0.16349; 0.16172; 0.16005; 0.15847; 0.15697; 0.15554; 0.15419; 0.1529; 0.15168; 0.15051; 0.14938; 0.14831; 0.14728; 0.14629; 0.14534; 0.14443; 0.14355; 0.14268; 0.14187; 0.14108; 0.14031; 0.13958; 0.13887; 0.13818; 0.13751; 0.13685; 0.13622; 0.1356; 0.13501; 0.13443; 0.13386; 0.13332; 0.13278; 0.13227; 0.13176; 0.13126; 0.13078; 0.1303; 0.12984; 0.12939; 0.12895; 0.12852; 0.12809; 0.12768; 0.12728; 0.12688; 0.12649; 0.12611; 0.12573; 0.12537; 0.12501; 0.12466; 0.12431; 0.12397; 0.12363; 0.1233; 0.12298; 0.12266; 0.12235; 0.12204; 0.12174; 0.12144; 0.12114; 0.12085; 0.12056; 0.12028; 0.12; 0.11973; 0.11946; 0.11919; 0.11893; 0.11867; 0.11842; 0.11817; 0.11792; 0.11767; 0.11743; 0.11719; 0.11695; 0.11672; 0.11649; 0.11626; 0.11603; 0.1158; 0.11558; 0.11536; 0.11515; 0.11494; 0.11473; 0.11452; 0.11431; 0.1141; 0.11389; 0.11369; 0.11349; 0.1133; 0.1131; 0.11291; 0.11272; 0.11253; 0.11232; 0.11214; 0.11195; 0.11177; 0.11159; 0.11141; 0.11124; 0.11106; 0.11088; 0.11071; 0.11054; 0.11037; 0.1102; 0.11003; 0.10987; 0.1097; 0.10954; 0.10937];

native = load(fullfile(data_dir, 'wavexisao_nopar.mat'), 'lambda_array', 'T_down_array');
RETICOLO_full = native.T_down_array;

LIVIA_extracted = data.T_fixed_dx(data.theta_indices(2), :);

%% Geometry for panel a (nm)
Period = 900;       w_tooth = 300;      h_grating = 200; 
slant_theta = 60;   gap = 300;
dx_slant = h_grating / tand(slant_theta);

y_bot_base = -gap/2 - h_grating;  y_bot_top  = -gap/2;
y_top_bot  = gap/2;               y_top_base = gap/2 + h_grating;

%% Figure layout
fig = figure('Color', 'w', 'Position', [100, 100, 1500, 850], 'Name', 'FW-BIC Final Rendering');
t = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

%% Panel a: Device schematic and COMSOL field
t_a = tiledlayout(t, 1, 2, 'TileSpacing', 'tight', 'Padding', 'none');
t_a.Layout.Tile = 1; t_a.Layout.TileSpan = [1 1];

ax_a1 = nexttile(t_a, 1);
hold(ax_a1, 'on'); axis(ax_a1, 'equal'); axis(ax_a1, 'off');

% Schematic dimensions are for illustration.
hd = 400;
wd = 350;
gapd = 300;
thetad = 60;
dxd = hd / tand(thetad); 
shift_d = 450;

y_b_base = -gapd/2 - hd;  % -550
y_b_top = -gapd/2;        % -150
y_t_bot = gapd/2;         %  150
y_t_base = gapd/2 + hd;   %  550

fill(ax_a1, [-2000 3000 3000 -2000], [y_b_base y_b_base y_t_base y_t_base], [0.8 0.95 1.0], 'EdgeColor', 'none');
fill(ax_a1, [-2000 3000 3000 -2000], [-3000 -3000 y_b_base y_b_base], [0.9 0.9 0.9], 'EdgeColor', 'none');
fill(ax_a1, [-2000 3000 3000 -2000], [y_t_base y_t_base 3000 3000], [0.9 0.9 0.9], 'EdgeColor', 'none');

center_x = shift_d/2;
text(ax_a1, center_x + 500, 0, 'Liquid', 'Color', [0 0.3 0.6], 'FontSize', 14, 'FontWeight', 'bold');
text(ax_a1, center_x, y_b_base - 200, 'Substrate', 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');
text(ax_a1, center_x, y_t_base + 200, 'Top Substrate', 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');

pts_bot_d = [wd/2, y_b_base; wd/2 - dxd, y_b_top; -wd/2 - dxd, y_b_top; -wd/2, y_b_base];
fill(ax_a1, pts_bot_d(:,1), pts_bot_d(:,2), [0.15 0.45 0.75], 'EdgeColor', 'k', 'LineWidth', 1.5);

pts_top_d = [wd/2 - dxd + shift_d, y_t_bot; wd/2 + shift_d, y_t_base; -wd/2 + shift_d, y_t_base; -wd/2 - dxd + shift_d, y_t_bot];
fill(ax_a1, pts_top_d(:,1), pts_top_d(:,2), [0.15 0.45 0.75], 'EdgeColor', 'k', 'LineWidth', 1.5);

% \Delta x
plot(ax_a1, [0 shift_d], [0 0], 'k|-', 'LineWidth', 1.5, 'MarkerSize', 8);
text(ax_a1, shift_d/2, 100, '\Delta x', 'FontSize', 16, 'HorizontalAlignment', 'center', 'Interpreter', 'tex', 'FontWeight', 'bold');

plot(ax_a1, [-wd/2, -wd/2 - 400], [y_b_base, y_b_base], 'k--', 'LineWidth', 1.2);
arc_r = 200; 
arc_t = linspace(180 - thetad, 180, 20); 
plot(ax_a1, -wd/2 + arc_r*cosd(arc_t), y_b_base + arc_r*sind(arc_t), 'k', 'LineWidth', 1.5);
text(ax_a1, -wd/2 + 1.4*arc_r*cosd(180 - thetad/2), y_b_base + 1.4*arc_r*sind(180 - thetad/2), '\theta', 'FontSize', 16, 'Interpreter', 'tex');

% \Lambda
plot(ax_a1, [-wd/2, -wd/2 + 900], [y_b_base - 500, y_b_base - 500], 'k|-', 'LineWidth', 1.5, 'MarkerSize', 8);
text(ax_a1, -wd/2 + 450, y_b_base - 650, '\Lambda', 'FontSize', 16, 'HorizontalAlignment', 'center', 'Interpreter', 'tex');

set(ax_a1, 'XLim', [-1000 1700], 'YLim', [-1500 1500]);
text(ax_a1, -0.1, 1.05, 'a', 'Units', 'normalized','FontName','Arial', 'FontSize', 22, 'FontWeight', 'bold');

ax_a2 = nexttile(t_a, 2);
load(fullfile(reference_dir, 'XYZ.mat'), 'XYE');

%%
x_comsol = XYE(:,1);
y_comsol = XYE(:,2);
E_comsol = XYE(:,3);


res_interp = 400; 
x_lin = linspace(min(x_comsol), max(x_comsol), res_interp);
y_lin = linspace(min(y_comsol), max(y_comsol), res_interp*2); 
[X1, Y1] = meshgrid(x_lin, y_lin);
F_interp = scatteredInterpolant(x_comsol, y_comsol, E_comsol, 'natural', 'none');
E1 = F_interp(X1, Y1);

X_full = [X1 - Period, X1, X1 + Period]; Y_full = [Y1, Y1, Y1]; E_full = [E1, E1, E1];

hold(ax_a2, 'on');
imagesc(ax_a2, X_full(1, :), Y_full(:, 1), E_full, 'AlphaData', isfinite(E_full));
set(ax_a2, 'YDir', 'normal');

pts_bot = [w_tooth/2, y_bot_base; w_tooth/2 - dx_slant, y_bot_top; -w_tooth/2 - dx_slant, y_bot_top; -w_tooth/2, y_bot_base; w_tooth/2, y_bot_base];
pts_top_center = [w_tooth/2 - dx_slant, y_top_bot; w_tooth/2, y_top_base; -w_tooth/2, y_top_base; -w_tooth/2 - dx_slant, y_top_bot; w_tooth/2 - dx_slant, y_top_bot];
for i = -1:1
    plot(ax_a2, pts_bot(:,1) + i*Period, pts_bot(:,2), 'k', 'LineWidth', 1.5, 'LineJoin', 'round');
    plot(ax_a2, pts_top_center(:,1) + i*Period, pts_top_center(:,2), 'k', 'LineWidth', 1.5, 'LineJoin', 'round');
end
plot(ax_a2, [min(X_full(:)) max(X_full(:))], [y_bot_base y_bot_base], 'k', 'LineWidth', 1.5);
plot(ax_a2, [min(X_full(:)) max(X_full(:))], [y_top_base y_top_base], 'k', 'LineWidth', 1.5);

colormap(ax_a2, "turbo"); axis(ax_a2, 'equal'); axis(ax_a2, 'off');

xlim(ax_a2, [-Period*1.5, Period*1.5]); 
ylim(ax_a2, [-1500, 1500]); 
text(ax_a2, 0.5, 1.03, {'COMSOL: Norm(E)', '\lambda = 1610 nm'}, ...
    'Units', 'normalized', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
    'FontSize', 15, 'FontWeight', 'bold');
%% Panel b: Reference spectra
ax_b = nexttile(t, 2); 
hold(ax_b, 'on');
h_comsol = plot(ax_b, lambda_1d_full, COMSOL_full, '-', 'Color', [0.15 0.15 0.15], 'LineWidth', 2.5); 
h_reticolo = plot(ax_b, native.lambda_array, RETICOLO_full, '-', 'Color', [0.85 0.25 0.15], 'LineWidth', 1.5);
h_livia = plot(ax_b, lambda_array3D, LIVIA_extracted, 'o', 'MarkerSize', 6, 'MarkerEdgeColor', [0.1 0.4 0.8], 'MarkerFaceColor', 'w', 'LineWidth', 1.5, 'MarkerIndices', 1:2:length(lambda_array3D));
box(ax_b, 'on'); grid(ax_b, 'on'); set(ax_b, 'FontSize', 12, 'LineWidth', 1.2, 'GridAlpha', 0.3);
xlabel(ax_b, 'Wavelength \lambda (\mu m)', 'FontSize', 14, 'FontWeight', 'bold'); ylabel(ax_b, 'Transmission', 'FontSize', 14, 'FontWeight', 'bold');
legend(ax_b, [h_comsol, h_reticolo, h_livia], {'COMSOL FEM', 'RETICOLO', 'LIVIA (Tensor)'}, 'Location', 'northeast', 'FontSize', 11, 'Box', 'off');
xlim(ax_b, [1.600 1.620]); ylim(ax_b, [0 1.05]); 
text(ax_b, -0.15, 1.05, 'b', 'Units', 'normalized','FontName','Arial', 'FontSize', 22, 'FontWeight', 'bold');

%% Panel c: Wavelength–offset slice at 60 degrees
ax_c = nexttile(t, 3); 
slice_60 = squeeze(data.T_fixed_theta(2, :, :)).';
imagesc(ax_c, dx_array, lambda_array3D, slice_60);
set(ax_c, 'YDir', 'normal', 'FontSize', 12, 'LineWidth', 1.2);
shading(ax_c, 'flat'); colormap(ax_c, "turbo"); clim(ax_c, [0 1]);
xlabel(ax_c, '\Delta x (\mu m)', 'FontSize', 14, 'FontWeight', 'bold'); ylabel(ax_c, 'Wavelength \lambda (\mu m)', 'FontSize', 14, 'FontWeight', 'bold');
text(ax_c, 0.05, 0.92, sprintf('\\theta = %.1f^\\circ', theta_array(data.theta_indices(2))), 'Units', 'normalized', 'Color', 'w', 'FontSize', 16, 'FontWeight', 'bold');
text(ax_c, -0.15, 1.05, 'c', 'Units', 'normalized','FontName','Arial', 'FontSize', 22, 'FontWeight', 'bold');
colorbar
%% Panel d: Three-dimensional transmission cuts
ax_d = nexttile(t, 4); ax_d.Layout.TileSpan = [1 2]; 

theta_extent = theta_array([1 end]);
lambda_extent = lambda_array3D([1 end]);
dx_extent = dx_array([1 end]);
texture_options = {'FaceColor', 'texturemap', 'EdgeColor', 'none', 'FaceAlpha', 0.6};
[Y, Z] = meshgrid(theta_extent, lambda_extent);
surf(ax_d, dx_array(data.dx_index) * ones(size(Y)), Y, Z, ...
    'CData', data.T_fixed_dx.', texture_options{:});
hold(ax_d, 'on');
[X, Z] = meshgrid(dx_extent, lambda_extent);
for i = 1:numel(data.theta_indices)
    theta_cut = theta_array(data.theta_indices(i));
    surf(ax_d, X, theta_cut * ones(size(X)), Z, ...
        'CData', squeeze(data.T_fixed_theta(i, :, :)).', texture_options{:});
end
[X, Y] = meshgrid(dx_extent, theta_extent);
surf(ax_d, X, Y, lambda_array3D(data.lambda_index) * ones(size(X)), ...
    'CData', data.T_fixed_lambda, texture_options{:});
hold(ax_d, 'off');
colormap(ax_d, "turbo"); clim(ax_d, [0 1]);
xlabel(ax_d, '\Delta x (\mu m)', 'FontSize', 14, 'FontWeight', 'bold'); ylabel(ax_d, '\theta (deg)', 'FontSize', 14, 'FontWeight', 'bold'); zlabel(ax_d, '\lambda (\mu m)', 'FontSize', 14, 'FontWeight', 'bold');
view(ax_d, 105, 18); grid(ax_d, 'on'); set(ax_d, 'FontSize', 12, 'LineWidth', 1.2); camproj(ax_d, 'perspective');
cb_d = colorbar(ax_d); ylabel(cb_d, 'Transmission', 'FontSize', 14, 'FontWeight', 'bold');
text(ax_d, -0.05, 1.05, 'd', 'Units', 'normalized','FontName','Arial', 'FontSize', 22, 'FontWeight', 'bold');

%% Panel e: Slant–offset slice at 1.610 um
ax_e = nexttile(t, 6); 
idx_l161 = data.lambda_index;
slice_161 = data.T_fixed_lambda;
imagesc(ax_e, dx_array, theta_array, slice_161);
set(ax_e, 'YDir', 'normal', 'FontSize', 12, 'LineWidth', 1.2);
shading(ax_e, 'flat'); colormap(ax_e, "turbo"); clim(ax_e, [0 1]);
xlabel(ax_e, '\Delta x (\mu m)', 'FontSize', 14, 'FontWeight', 'bold'); ylabel(ax_e, 'Slant Angle \theta (deg)', 'FontSize', 14, 'FontWeight', 'bold');
text(ax_e, 0.05, 0.92, sprintf('\\lambda = %.3f \\mu m', lambda_array3D(idx_l161)), 'Units', 'normalized', 'Color', 'w', 'FontSize', 16, 'FontWeight', 'bold');
colorbar
text(ax_e, -0.15, 1.05, 'e', 'Units', 'normalized', 'FontSize', 22,'FontName','Arial', 'FontWeight', 'bold');

exportgraphics(fig, fullfile(output_dir, 'Fig6_REALL.pdf'), 'ContentType', 'vector');
