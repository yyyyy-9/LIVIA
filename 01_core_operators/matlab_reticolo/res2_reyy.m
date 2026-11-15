function [s_total, beth, betb, sh, sb] = res2_reyy(aa, PROFIL, parm)
% Modified excerpt from RETICOLO V10 res2.m (J.-P. Hugonin & P. Lalanne, DOI: 10.5281/zenodo.14631951);
% provided under CC BY 4.0 with attribution (see THIRD_PARTY.md).
% Modification: omits exterior port truncation (rettronc) to return the full four-block S-matrix.
% Requires RETICOLO on the MATLAB path.

aa = retio(aa);

if nargin < 3
    parm = res0;
end

if ~iscell(PROFIL{1})
    PROFIL = {PROFIL};
end

ih = PROFIL{1}{2}(1);
ib = PROFIL{end}{2}(end);

a = aa{1}; init = aa{2}; n = aa{3}; UN = aa{4}; D = aa{5}; beta0 = aa{6}; sym = aa{7};

if init{end}.sog
    retss(parm.res2.retss);
else
    retss(parm.res2.retgg);
end

TAB = zeros(0, 2);

if parm.res2.cale == 1
    [sb, bas, betb, cb, anglb] = retb(init, a{ib}, -parm.res2.tolb);
end

if parm.res2.cals == 1
    s = sb;
end

for module = size(PROFIL, 2):-1:1
    sequence = PROFIL{module}{2};
    ntextures = length(sequence);
    HAUTEURS = PROFIL{module}{1};
    if size(PROFIL{module}, 2) >= 3
        fois = PROFIL{module}{3};
    else
        fois = 1;
    end

    if fois == 1
        for ii = ntextures:-1:1
            TAB = [[HAUTEURS(ii), sequence(ii)]; TAB];
            if HAUTEURS(ii) ~= 0 & parm.res2.cals == 1
                s = retss(retc(a{sequence(ii)}, HAUTEURS(ii) / UN), s);
            end
        end
    else
        premier = 1; TABB = [];
        for ii = ntextures:-1:1
            TABB = [[HAUTEURS(ii), sequence(ii)]; TABB];
            if HAUTEURS(ii) ~= 0 & parm.res2.cals == 1
                if premier == 1; premier = 0;
                    ss = retc(a{sequence(ii)}, HAUTEURS(ii) / UN);
                else
                    ss = retss(retc(a{sequence(ii)}, HAUTEURS(ii) / UN), ss);
                end
            end
        end

        if premier == 0 & parm.res2.cals == 1
            s = retss(retsp(ss, fois), s);
        end
        TAB = [repmat(TABB, fois, 1); TAB];
    end
end

if parm.res2.cale ~= 1
    s_total = [];
    beth = [];
    betb = [];
    sh = [];
    sb = [];
    return;
end

[sh, haut, beth, ch, anglh] = retb(init, a{ih}, parm.res2.tolh);

if parm.res2.cals == 1
    s = retss(sh, s);
else
    s = [];
end

s_total = s;

end
