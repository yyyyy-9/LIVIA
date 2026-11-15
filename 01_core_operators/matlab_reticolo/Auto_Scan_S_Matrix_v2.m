function Result = Auto_Scan_S_Matrix_v2(S_mat, aa, profil)
% Extract propagating 1D diffraction orders from a full RETICOLO S-matrix.
% Each table row is [order, efficiency, real(amplitude), imag(amplitude)].

    dim = size(S_mat, 1) / 2;
    center_row = (dim - 1) / 2 + 1;

    ih = profil{1}{2}(1);
    ib = profil{end}{2}(end);

    P_top = aa{1}{ih}{1};
    d_top = aa{1}{ih}{5};
    P_bot = aa{1}{ib}{1};
    d_bot = aa{1}{ib}{5};

    Map_Top = detect_propagating_modes(P_top, d_top, center_row);
    Map_Bot = detect_propagating_modes(P_bot, d_bot, center_row);

    idx_top_0 = Map_Top.indices(Map_Top.orders == 0);
    idx_bot_0 = Map_Bot.indices(Map_Bot.orders == 0);
    if isempty(idx_top_0) || isempty(idx_bot_0)
        warning('No propagating zero order was found at one or both ports.');
    end

    if ~isempty(idx_top_0)
        col_in = idx_top_0 + dim;
        Result.T_down = extract_order_table(S_mat, col_in, Map_Bot, dim);
        Result.R_up   = extract_order_table(S_mat, col_in, Map_Top, 0);
    end

    if ~isempty(idx_bot_0)
        col_in = idx_bot_0;
        Result.T_up   = extract_order_table(S_mat, col_in, Map_Top, 0);
        Result.R_down = extract_order_table(S_mat, col_in, Map_Bot, dim);
    end
end

function Map = detect_propagating_modes(P, d, center_row)
    [~, max_rows] = max(abs(P), [], 1);
    keep = abs(real(d(:).')) <= 1e-3;
    valid_indices = find(keep);
    valid_orders  = max_rows(keep) - center_row;

    [sorted_orders, sort_idx] = sort(valid_orders);
    Map.orders  = sorted_orders;
    Map.indices = valid_indices(sort_idx);
end

function table = extract_order_table(S, input_col, Map_Output, row_offset)
    rows = Map_Output.indices(:) + row_offset;
    vals = S(rows, input_col);
    table = [Map_Output.orders(:), abs(vals).^2, real(vals), imag(vals)];
end
