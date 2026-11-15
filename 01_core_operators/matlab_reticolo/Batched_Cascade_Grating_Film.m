function [S11_tot, S12_tot, S21_tot, S22_tot] = Batched_Cascade_Grating_Film( ...
    Sg11, Sg12, Sg21, Sg22, Sf11, Sf12, Sf21, Sf22)
% Four-block Redheffer cascade of a fixed grating S-matrix (N-by-N)
% with a batch of diagonal film S-matrices (B-by-N). Outputs are N-by-N-by-B.

    [num_batches, N] = size(Sf11);

    Sf11_row = reshape(Sf11.', [N, 1, num_batches]);
    Sf21_row = reshape(Sf21.', [N, 1, num_batches]);
    Sf22_row = reshape(Sf22.', [N, 1, num_batches]);

    Sf11_col = reshape(Sf11.', [1, N, num_batches]);
    Sf12_col = reshape(Sf12.', [1, N, num_batches]);

    I_3D = repmat(eye(N), 1, 1, num_batches);
    Z       = pageinv(I_3D - Sg22 .* Sf11_col);
    Z_prime = pageinv(I_3D - Sf11_row .* Sg22);

    Sf11_Z = Sf11_row .* Z;
    Z_prime_Sf12 = Z_prime .* Sf12_col;

    S11_tot = Sg11 + pagemtimes(Sg12, pagemtimes(Sf11_Z, Sg21));
    S21_tot = Sf21_row .* pagemtimes(Z, Sg21);
    S12_tot = pagemtimes(Sg12, Z_prime_Sf12);
    S22_tot = Sf22_row .* eye(N) + Sf21_row .* pagemtimes(Sg22, Z_prime_Sf12);
end
