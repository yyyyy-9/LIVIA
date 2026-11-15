% Rebuild the four panels of Figure 2 from the survey summary and PDPS CDFs.
clear; clc; close all;

base = fileparts(mfilename('fullpath'));
outputs_dir = fullfile(base, 'outputs');
data_dir = outputs_dir;
if strcmp(getenv('LIVIA_PLOT_GENERATED'), '1')
    data_dir = fullfile(outputs_dir, 'generated');
    outputs_dir = data_dir;
end
D = jsondecode(fileread(fullfile(data_dir, 'data.json')));

c_80D  = [214, 39, 40]/255;    c_ctrl = [31, 119, 180]/255;
c_top  = [44, 160, 44]/255;    c_dr   = [255, 152, 40]/255; 
c_both = [34, 139, 34]/255;    c_none = [228, 228, 228]/255; 
c_wide = [0.88, 0.88, 0.88];

set(0, 'DefaultAxesFontSize', 10, 'DefaultAxesFontName', 'Arial', ...
       'DefaultAxesLineWidth', 1.1, 'DefaultAxesTickDir', 'in');

fig = figure('Color', 'w', 'Position', [100, 100, 1050, 680]);
t = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

%% (a) Unconditional P(R0>=64%)
axA = nexttile(1, [1 2]); hold(axA, 'on');
narrow = [D.x80D.p64; D.x32D_ctrl.p64; D.x32D_top.p64];
wcount = [D.wide.x80D; D.wide.x32D_ctrl; D.wide.x32D_top];
wide = max(wcount/67108864*100, 1e-6);

b = bar(axA, 1:3, [narrow wide], 0.75, 'grouped', 'EdgeColor', 'k', 'LineWidth', 0.9);
b(1).FaceColor = 'flat'; b(1).CData = [c_80D; c_ctrl; c_top];
b(2).FaceColor = c_wide;

set(axA, 'YScale', 'log', 'YLim', [5e-7, 25], 'XTick', 1:3, ...
    'XTickLabel', {'80D Independent', '32D Graded', '32D Top'}, 'Box', 'on');
ylabel(axA, 'Unconditional P(R_0 \geq 64%) [%]', 'FontWeight', 'bold'); 
grid(axA, 'on'); set(axA, 'YGrid', 'on', 'YMinorGrid', 'on', 'XGrid', 'off', 'GridAlpha', 0.25, 'MinorGridAlpha', 0.1);

text(axA, 0.85, narrow(1)*1.4, sprintf('%.3f%%', narrow(1)), 'FontWeight', 'bold', 'FontSize', 10, 'HorizontalAlignment', 'center');
text(axA, 1.85, narrow(2)*1.4, sprintf('%.2f%%', narrow(2)), 'FontWeight', 'bold', 'FontSize', 10, 'HorizontalAlignment', 'center');
text(axA, 2.85, narrow(3)*1.4, sprintf('%.2f%%', narrow(3)), 'FontWeight', 'bold', 'FontSize', 10, 'HorizontalAlignment', 'center');
text(axA, 1.15, 2e-6, '0', 'Color', c_80D, 'FontWeight', 'bold', 'FontSize', 10, 'HorizontalAlignment', 'center');
text(axA, 3.15, 2e-6, '0', 'Color', c_80D, 'FontWeight', 'bold', 'FontSize', 10, 'HorizontalAlignment', 'center');

legend(axA, b, {'Narrow bound [2.50, 4.84] nm', 'Wide bound [2.50, 7.92] nm'}, 'Location', 'northwest', 'FontSize', 9, 'Box', 'off');
text(axA, 0.0, 1.04, 'a', 'Units', 'normalized', 'FontWeight', 'bold', 'FontSize', 13, ...
     'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');

%% (b) PDPS Conditional CDF (ground truth, from full_pool direct count)
axB = nexttile(3, [1 1]); hold(axB, 'on');

% Read PDPS CDF from ground truth CSV
ctrl_cdf = readmatrix(fullfile(data_dir, 'pdps_cdf_32D_ctrl.csv'));
top_cdf  = readmatrix(fullfile(data_dir, 'pdps_cdf_32D_top.csv'));

plot(axB, ctrl_cdf(:,1), ctrl_cdf(:,2), '-', 'Color', c_ctrl, 'LineWidth', 2.5);
plot(axB, top_cdf(:,1),  top_cdf(:,2),  '-', 'Color', c_top,  'LineWidth', 2.5);
xline(axB, 7, '--k', 'LineWidth', 1.2, 'Alpha', 0.4);

% Ground truth PDPS<7 from data.json
ctrl_p7 = D.x32D_ctrl.p_pdps_lt7;   % 59.4%
top_p7  = D.x32D_top.p_pdps_lt7;    % 9.2%

yline(axB, ctrl_p7, ':', 'Color', c_ctrl, 'LineWidth', 1.0, 'Alpha', 0.6);
yline(axB, top_p7,  ':', 'Color', c_top,  'LineWidth', 1.0, 'Alpha', 0.6);

text(axB, 7.5, ctrl_p7 + 2, sprintf('%.1f%%', ctrl_p7), 'Color', c_ctrl, 'FontWeight', 'bold', 'FontSize', 10, 'VerticalAlignment', 'bottom');
text(axB, 7.5, top_p7 - 2,  sprintf('%.1f%%', top_p7),  'Color', c_top,  'FontWeight', 'bold', 'FontSize', 10, 'VerticalAlignment', 'top');

set(axB, 'Box', 'on');
xlabel(axB, 'PDPS (deg)', 'FontWeight', 'bold'); ylabel(axB, 'CDF [%] (R_0 \geq 64%)', 'FontWeight', 'bold');
xlim(axB, [0, 20]); ylim(axB, [0, 105]); grid(axB, 'on'); set(axB, 'GridAlpha', 0.25);
legend(axB, {'32D Graded', '32D Top'}, 'Location', 'southeast', 'FontSize', 9, 'Box', 'off');
text(axB, 0.0, 1.04, 'b', 'Units', 'normalized', 'FontWeight', 'bold', 'FontSize', 13, ...
     'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');

%% Data for stacked bars
aks = {'x80D', 'x32D_ctrl', 'x32D_top'};
S = zeros(3,4);
for i=1:3
    S(i,1)=D.(aks{i}).stack.both; S(i,2)=D.(aks{i}).stack.dr_only;
    S(i,3)=D.(aks{i}).stack.pdps_only; S(i,4)=D.(aks{i}).stack.neither;
end

%% (c) Stacked bar (0-100%)
axC = nexttile(4, [1 2]); hold(axC, 'on');
bC = bar(axC, 1:3, S, 0.55, 'stacked', 'EdgeColor', 'k', 'LineWidth', 0.9);
bC(1).FaceColor = c_both; bC(2).FaceColor = c_dr;
bC(3).FaceColor = c_ctrl; bC(4).FaceColor = c_none;

set(axC, 'YLim', [0, 115], 'XTick', 1:3, 'XTickLabel', {'80D Indep.', '32D Graded', '32D Top'}, 'Box', 'on');
ylabel(axC, 'Fraction of R_0 \geq 64% designs [%]', 'FontWeight', 'bold');
grid(axC, 'on'); set(axC, 'YGrid', 'on', 'XGrid', 'off', 'GridAlpha', 0.25);

for i=1:3
    cy = 0;
    for j = 1:4
        v = S(i,j);
        if j == 3 && S(i,1) > 0 && S(i,1) < 5
            v = S(i,1) + S(i,3);
        end
        if v > 5
            text(axC, i, cy + v/2, sprintf('%.1f%%', v), ...
                'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10, 'Color', 'k');
        end
        cy = cy + S(i,j);
    end
end

legend(axC, bC, {'Both (<10%, <7^\circ)', '\DeltaR<10% only', 'PDPS<7^\circ only', 'Neither'}, ...
    'NumColumns', 4, 'FontSize', 9, 'Location', 'north', 'Box', 'off');
text(axC, 0.0, 1.04, 'c', 'Units', 'normalized', 'FontWeight', 'bold', 'FontSize', 13, ...
     'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');

%% (d) Magnified view (0-0.6%)
axD = nexttile(6, [1 1]); hold(axD, 'on');
bD = bar(axD, 1:3, S, 0.55, 'stacked', 'EdgeColor', 'k', 'LineWidth', 0.9);
bD(1).FaceColor = c_both; bD(2).FaceColor = c_dr;
bD(3).FaceColor = c_ctrl; bD(4).FaceColor = c_none;

set(axD, 'YLim', [0, 0.6], 'XTick', 1:3, 'XTickLabel', {'80D', '32D Gr.', '32D Top'}, 'Box', 'on');
ylabel(axD, 'Fraction [%] (Micro-Scale)', 'FontWeight', 'bold');
grid(axD, 'on'); set(axD, 'YGrid', 'on', 'XGrid', 'off', 'GridAlpha', 0.25);

for i=1:3
    cy=0;
    for j=1:2
        v = S(i,j);
        if v >= 0.1
            t_color = 'k';
            if j == 1, t_color = 'w'; end
            text(axD, i, cy + v/2, sprintf('%.2f%%', v), ...
                'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10, 'Color', t_color);
        end
        cy = cy + v;
    end
end

text(axD, 0.0, 1.04, 'd', 'Units', 'normalized', 'FontWeight', 'bold', 'FontSize', 13, ...
     'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');

%% Export
exportgraphics(fig, fullfile(outputs_dir, 'Figure_Mechanism.pdf'), 'ContentType', 'vector');
fprintf('Figure_Mechanism.pdf exported\n');
