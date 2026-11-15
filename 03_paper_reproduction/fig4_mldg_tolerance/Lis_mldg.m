% RETICOLO diffraction efficiencies at the nominal MLDG geometry.

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
huplayer = [hc, 0.1003, 0.2687, 0.1353, 0.2000, 0.1878, 0.2226, 0.1212, 0.3479, h_H]*1;

num_pairs     = 14;

d_list        = [huplayer, repmat([h_L, h_H], 1, num_pairs)];
n_stack_inner = repmat([4, 5], 1, num_pairs+5);

k0          = nAir*sin(theta*pi/180);

textures{1} = nAir;
textures{2} = n_sub;
textures{3} = {[-L/2, L/2], [nAir, n_SiO2]};

textures{4} = n_SiO2;
textures{5} = n_Ta2O5;

profil = {[1, hg, d_list, 1], [1, 3, n_stack_inner, 2]};

alpha0_list = linspace(0, 10.0, 1000);

tic

parm       = res0(1);
[aa_te, ~] = res1(wavelength, D_period, textures, nn, k0, parm);
parm       = res0(-1);
[aa_tm, ~] = res1(wavelength, D_period, textures, nn, k0, parm);

ef_te      = res2(aa_te, profil);
ef_tm      = res2(aa_tm, profil);

efficiency_ter      = ef_te.inc_top_reflected.efficiency;
efficiency_tmr      = ef_tm.inc_top_reflected.efficiency;

orders_ter    = ef_te.inc_top_reflected.order;
orders_tmr    = ef_tm.inc_top_reflected.order;

parm.res3.cale = []; 
x = linspace(-D_period/2*1000, D_period/2*1000, 2001)*1e-3;
y = 0; 
[e, z, index] = res3(x, y, aa_tm, profil, [], parm);
figure;
pcolor(x, z, real(squeeze(index)));
shading flat;
axis equal tight;
colorbar;
xlabel('x');
ylabel('z');

figure

plot(orders_ter, efficiency_ter, 'or-', orders_tmr, efficiency_tmr, 'ob-')

legend('TE','TM')
xlabel('Order')
ylabel('Diffraction Efficiency')
grid on

xlim([min(orders_ter)-1 max(orders_ter)+1])
xticks(min(orders_ter)-1:max(orders_ter)+1)

