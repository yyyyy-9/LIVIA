% Figure 4. Archived full-grid efficiencies and timing are the default.
% Set LIVIA_PLOT_GENERATED=1 to use the four locally regenerated MAT files.
clear; clc; close all;

base = fileparts(mfilename('fullpath'));
output_dir = fullfile(base, 'outputs');
data_dir = fullfile(base, 'reference');
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

reference = load(fullfile(data_dir, 'RETICOLOres.mat'), ...
    'Eff_TE2D_RETICOLO', 'Eff_TM2D_RETICOLO', 'T_H4', 'T_L4', 'time_reticolo');
livia = load(fullfile(data_dir, 'Batched_Cascade2.mat'), ...
    'EffTE_2D', 'EffTM_2D', 'T_H4', 'T_L4', 'Batched_time');
T_H4 = livia.T_H4;
T_L4 = livia.T_L4;
assert(isequal(T_H4, reference.T_H4) && isequal(T_L4, reference.T_L4), ...
    'LIVIA:ThicknessGridMismatch', 'The LIVIA and RETICOLO thickness grids differ.');
EffTE_2D = livia.EffTE_2D;
EffTM_2D = livia.EffTM_2D;
Eff_TE2D_RETICOLO = reference.Eff_TE2D_RETICOLO;
Eff_TM2D_RETICOLO = reference.Eff_TM2D_RETICOLO;
if use_generated
    full = load(fullfile(data_dir, 'RETICOLOresallorder.mat'), 'time_reticoloall', 'T_H4', 'T_L4');
    cpu = load(fullfile(data_dir, 'Batched_Cascade.mat'), 'Batched_time', 'T_H4', 'T_L4');
    assert(isequal(T_H4, full.T_H4, cpu.T_H4) && isequal(T_L4, full.T_L4, cpu.T_L4), ...
        'LIVIA:ThicknessGridMismatch', 'The four timing runs must use the same thickness grid.');
    times = [full.time_reticoloall, reference.time_reticolo, cpu.Batched_time, livia.Batched_time];
else
    times = [1623.1, 786.3, 104.5, 41.4];
end
set(0, 'DefaultAxesFontName', 'Arial'); set(0, 'DefaultTextFontName', 'Arial');
err_TE = abs(EffTE_2D - Eff_TE2D_RETICOLO);
err_TM = abs(EffTM_2D - Eff_TM2D_RETICOLO);
log_err_TE = log10(err_TE + 1e-16);
log_err_TM = log10(err_TM + 1e-16);

figure('Name', 'Performance and Fidelity Comparison', 'Color', 'w', 'Position', [100, 100, 1650, 780]);
subplot(2, 3, 1);
contourf(T_H4, T_L4, EffTE_2D, 100, 'LineStyle', 'none'); hold on;
contour(T_H4, T_L4, EffTE_2D, [0.97 0.97], 'LineColor', 'w', 'LineWidth', 2);
colormap(gca, 'jet'); colorbar; caxis([0.8 1.0]);
xlabel('H4 Thickness (\mu m)'); ylabel('L4 Thickness (\mu m)');
title('Tensor SMM (TE Polarization)', 'FontSize', 12, 'FontWeight', 'bold');
text(-0.1, 1.15, 'a', 'Units', 'normalized', 'FontSize', 16, 'FontWeight', 'bold');

subplot(2, 3, 2);
contourf(T_H4, T_L4, EffTM_2D, 100, 'LineStyle', 'none'); hold on;
contour(T_H4, T_L4, EffTM_2D, [0.97 0.97], 'LineColor', 'w', 'LineWidth', 2);
colormap(gca, 'jet'); colorbar; caxis([0.8 1.0]);
xlabel('H4 Thickness (\mu m)'); ylabel('L4 Thickness (\mu m)');
title('Tensor SMM (TM Polarization)', 'FontSize', 12, 'FontWeight', 'bold');
text(-0.1, 1.15, 'b', 'Units', 'normalized', 'FontSize', 16, 'FontWeight', 'bold');

subplot(2, 3, 4);
imagesc(T_H4(1,:), T_L4(:,1), log_err_TE); set(gca, 'YDir', 'normal');
colormap(gca, 'parula'); colorbar; caxis([-14 -8]);
xlabel('H4 Thickness (\mu m)'); ylabel('L4 Thickness (\mu m)');
title('Log_{10} Absolute Error (TE)', 'FontSize', 12, 'FontWeight', 'bold');
text(-0.1, 1.15, 'c', 'Units', 'normalized', 'FontSize', 16, 'FontWeight', 'bold');

subplot(2, 3, 5);
imagesc(T_H4(1,:), T_L4(:,1), log_err_TM); set(gca, 'YDir', 'normal');
colormap(gca, 'parula'); colorbar; caxis([-14 -8]);
xlabel('H4 Thickness (\mu m)'); ylabel('L4 Thickness (\mu m)');
title('Log_{10} Absolute Error (TM)', 'FontSize', 12, 'FontWeight', 'bold');
text(-0.1, 1.15, 'd', 'Units', 'normalized', 'FontSize', 16, 'FontWeight', 'bold');

subplot(1, 3, 3);
methods = {'Full RETICOLO', 'Native RETICOLO', 'LIVIA CPU', 'LIVIA GPU'};
b = bar(1:4, times, 'FaceColor', 'flat');
b.CData = [0.6 0.1 0.1; 0.9 0.4 0.3; 0.0 0.4 0.7; 0.2 0.7 0.3];
ylim([0 1.16 * max(times)]);
set(gca, 'XTick', 1:4, 'XTickLabel', methods, 'FontSize', 11);
xtickangle(20);
ylabel('Cascade time (s)', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('Cascade throughput\n%d states across TE and TM', 2 * numel(EffTE_2D)), ...
    'FontSize', 12, 'FontWeight', 'bold');
grid on;
labels = {sprintf('%.1f s', times(1)), sprintf('%.1f s', times(2)), ...
    sprintf('%.1f s\n(%.1fx)', times(3), times(1)/times(3)), ...
    sprintf('%.1f s\n(%.1fx)', times(4), times(1)/times(4))};
text(1:4, times + 0.025 * max(times), labels, ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold');
text(-0.1, 1.05, 'e', 'Units', 'normalized', 'FontSize', 16, 'FontWeight', 'bold');

set(findall(gcf, '-property', 'FontName'), 'FontName', 'Arial');
exportgraphics(gcf, fullfile(output_dir, 'Figure4_MLDGv1.pdf'), 'ContentType', 'vector');
