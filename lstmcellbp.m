function cell = lstmcellbp(cell, e)
% LSTMBP performs backpropagation
%   cell = lstmbp(cell, e)
% cell:  the lstmcell to be execute
% e:     error get from upper layer
%%
    [m, n] = size(e);
    if(n ~= cell.outputlen || m ~= size(cell.x, 1))
        error('Error input lengh is not correspond with the lstm setup!')
    end
    dmi = zeros(m + 1, cell.outputlen);
    dmai = zeros(m + 1, cell.outputlen);
    dmf = zeros(m + 1, cell.outputlen);
    dmaf = zeros(m + 1, cell.outputlen);
    dmc = zeros(m + 1, cell.outputlen);
    dmac = zeros(m + 1, cell.outputlen);
    dmgac = zeros(m + 1, cell.outputlen);
    dmo = zeros(m + 1, cell.outputlen);
    dmao = zeros(m + 1, cell.outputlen);
    dmh = zeros(m + 1,cell.outputlen);
    dx = zeros(m + 1,cell.inputlen + 1);
 %% backpropagation through time
    for t = m : -1 :1
        % d_h(t) = e(t) + d_ai(t+1) * W_ih + d_af(t+1) * W_fh +
        % d_ac(t+1) * W_ch + d_ao(t+1) * W_oh
        dmh(t, :) = e(t, :) + dmai(t + 1, : ) * cell.W_ih + dmaf(t + 1, :) * cell.W_fh + dmao(t + 1, :) * cell.W_oh + dmac(t + 1, :) * cell.W_ch;
        
        %d_o(t) = d_h(t) * gc(t)
        dmo(t, :) = dmh(t, :) .* cell.mgc(t, :);
        %d_ao(t) = d_o(t) * delta'(ao(t))
        dmao(t, :) = dmo(t, :) .* active_func_d(cell.mo(t, :), cell.delta);
        
        % d_c(t) = d_h(t) .* o(t) * g'(c(t)) + d_ai(t+1) * W_ic + d_af(t+1) * W_fc + d_ao(t) * W_oc +
        % d_c(t+1) .* f(t)
        if t == m
            dmc(t, :) = dmh(t, :) .* cell.mo(t, :) .* active_func_d(cell.mgc(t, :), cell.g) + ...
                + dmai(t + 1, :) .* cell.W_ic' + dmaf(t + 1, :) .* cell.W_fc' + dmao(t, :) .* cell.W_oc'; % diagonal matrix represent by vertor
        else
            dmc(t, :) = dmh(t, :) .* cell.mo(t, :) .* active_func_d(cell.mgc(t, :), cell.g) + dmc(t + 1, :) .* cell.mf(t + 1, :) ...
                + dmai(t + 1, :) .* cell.W_ic' + dmaf(t + 1, :) .* cell.W_fc' + dmao(t, :) .* cell.W_oc'; % diagonal matrix represent by vertor
        end
        % d_ac(t) = d_c(t) .* i(t) * g'(ac(t))
        dmac(t, :) = dmc(t, :) .* active_func_d(cell.mgac(t, :), cell.g) .* cell.mi(t, :);
        % d_f(t) = d_c(t) .* c(t-1)
        if t > 1
            dmf(t, :) = dmc(t, :) .* cell.mc(t - 1, :);
            dmaf(t, :) = dmf(t, :) .* active_func_d(cell.mf(t, :), cell.delta);
        end
        % d_i(t) = d_c(t) .* gac(t)
        dmi(t, :) = dmc(t, :) .* cell.mgac(t, :);
        dmai(t, :) = dmi(t, :) .* active_func_d(cell.mi(t, :), cell.delta);
        % d_x(t) = d_ai(t) * W_ix + d_af(t) * W_fx + d_ao(t) * W_ox +
        % d_ac(t) * W_cx
        dx(t, :) = dmai(t, :) * cell.W_ix + dmaf(t, :) * cell.W_fx + dmao(t, :) * cell.W_ox + dmac(t, :) * cell.W_cx;
        
    end
%%    
    cell.e = e(1:m, :);
    cell.dmi = dmi(1:m, :);
    cell.dmai = dmai(1:m, :);
    cell.dmf = dmf(1:m, :);
    cell.dmaf = dmaf(1:m, :);
    cell.dmc = dmc(1:m, :);
    cell.dmac = dmac(1:m, :);
    cell.dmgac = dmgac(1:m, :);
    cell.dmo = dmo(1:m, :);
    cell.dmao = dmao(1:m, :);
    cell.dmh = dmh(1:m, :);
    cell.dx = dx(1:m, :);
%%
    
    cell.dW_ix = cell.dmai' * cell.x / m;
    cell.dW_ih = cell.dmai(2 : end, :)' * cell.mh(1 : end-1, :) / (m - 1);
    cell.dW_ic = (sum(cell.dmai(2 : end, :) .* cell.mc(1 : end-1, :)) / (m - 1))';
    
    cell.dW_fx = cell.dmaf' * cell.x / m;
    cell.dW_fh = cell.dmaf(2 : end, :)' * cell.mh(1 : end-1, :) / (m - 1);
    cell.dW_fc = (sum(cell.dmaf(2 : end, :) .* cell.mc(1 : end-1, :)) / (m - 1))';
    
    cell.dW_cx = cell.dmac' * cell.x / m;
    cell.dW_ch = cell.dmac(2 : end, :)' * cell.mh(1:end - 1, :) / (m - 1);
    
    cell.dW_ox = cell.dmao' * cell.x / m;
    cell.dW_oh = cell.dmao(2 : end, :)' * cell.mh(1 : end-1, :) / (m - 1);
    cell.dW_oc = (sum(cell.dmao .* cell.mc) / (m - 1))';
    
 
end