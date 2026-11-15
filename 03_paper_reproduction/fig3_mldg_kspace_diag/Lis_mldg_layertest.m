% RETICOLO thickness-scale sweep for Fig. 3c. All film layers scale together.

clc; 
clear; 

run(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'setup_paths.m'));

D_period   = 0.9;

num_orders = 100;
nn         = num_orders;
wavelength = 1.065;
theta      = asind(1.065/(2*D_period)); 

nAir  = 1.0;
n_sub = 1.51;

n_SiO2  = 1.46;            
n_Ta2O5 = 2.14;    

hg      = 1;
f       = 0.65;                  
L       = D_period*f;

h_H     = 0.1295;
h_L     = 0.1995;

htop    = 1.294;

hc       = htop - hg;
huplayer = [hc, 0.1003, 0.2687, 0.1353, 0.2000, 0.1878, 0.2226, 0.1212, 0.3479, h_H]*0;

num_pairs     = 20;

k0          = nAir*sin(theta*pi/180);

textures{1} = nAir;
textures{2} = n_sub;
textures{3} = {[-L/2, L/2], [nAir, n_SiO2]};

textures{4} = n_SiO2;
textures{5} = n_Ta2O5;

delete(gcp("nocreate"));
parpool('Threads',16)

tic

faclist = [0.98:0.001:0.984  0.985:0.00001:0.988 0.989:0.001:1.02 ];

parm       = res0(1);
[aa_te, ~] = res1(wavelength, D_period, textures, nn, k0, parm);
parm       = res0(-1);
[aa_tm, ~] = res1(wavelength, D_period, textures, nn, k0, parm);

% Figure 3c applies a common scale factor to the entire film stack.
d_nominal = repmat([h_L, h_H], 1, num_pairs);
parfor faci = 1:length(faclist)

    d_list = faclist(faci) * d_nominal;
    n_stack_inner = repmat([4, 5], 1, num_pairs);
    profil        = {[1, hg, d_list, 1], [1, 3, n_stack_inner, 2]};

    ef_te      = res2(aa_te, profil);
    ef_tm      = res2(aa_tm, profil);

    te_f1(faci)      = ef_te.inc_top_reflected.efficiency{-1};
    tm_f1(faci)      = ef_tm.inc_top_reflected.efficiency{-1};

    te_0(faci)       = ef_te.inc_top_reflected.efficiency{0};
    tm_0(faci)       = ef_tm.inc_top_reflected.efficiency{0};

end

 
output_dir = fullfile(fileparts(mfilename('fullpath')), 'outputs');
if ~isfolder(output_dir)
    mkdir(output_dir);
end
save(fullfile(output_dir, 'f065_relayer.mat'), 'faclist', 'te_f1', 'te_0', 'tm_f1', 'tm_0');

plot(faclist, te_f1, 'r-', faclist, te_0, 'b-')

return


