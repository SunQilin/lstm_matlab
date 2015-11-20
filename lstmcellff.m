function cell = lstmcellff(cell, x, y)
    %LSTMFF performs a feedforward pass
    %   lstmff(lstmcell, x, y)

    [m, n] = size(x);
    if(n ~= cell.inputlen + 1)
        error('Error!!!  Input lengh is not correspond with the lstm setup!')
    end
    mi = zeros(m, cell.outputlen);  % memory of the input gate 
    mai = zeros(m, cell.outputlen); % memory of the input gate's sumation 
    mf = zeros(m , cell.outputlen); % memory of the forget gate
    maf = zeros(m, cell.outputlen); % memory of the forget gate's sumation
    mc = zeros(m, cell.outputlen);  % memory of the cell
    mac = zeros(m, cell.outputlen); % memory of the cell's sumation
    mgac = zeros(m, cell.outputlen);    % memory of the activation computed by current input
    mo = zeros(m, cell.outputlen);  % memory of the output gate
    mao = zeros(m, cell.outputlen); % memory of the output gate's sumation
    mgc = zeros(m, cell.outputlen); % memory of the activation computed by current input and history
    mh = zeros(m, cell.outputlen);  % memory of the output(hidden layer output)

    %%comput the memory at time 1
    mai(1,:) =  x(1,:) * cell.W_ix';
    mi(1,:) = active_func(mai(1,:), cell.delta);

    maf(1,:) = x(1,:) * cell.W_fx';
    mf(1,:) = active_func(maf(1,:), cell.delta);

    mac(1,:) = x(1,:) * cell.W_cx';
    mgac(1,:) = active_func(mac(1,:), cell.g);
    mc(1,:) = mi(1,:) .*  mgac(1,:);

    mao(1,:) = x(1,:) * cell.W_ox' + mc(1,:) .* cell.W_oc';
    mo(1,:) = active_func(mao(1,:), cell.delta);
    
    mgc(1, :) = active_func(mc(1,:), cell.g);
    mh(1,:) = mo(1,:) .* mgc(1, :);

%% compute memory for each time    
    for t = 2 : m
        % a_i(t) = W_ix * x(t) + W_ih * h(t-1) + W_ic * c(t-1)
        mai(t,:) =  x(t,:) * cell.W_ix' + mh(t-1, :) * cell.W_ih' + mc(t-1, :) .* cell.W_ic';
        mi(t,:) = active_func(mai(t,:), cell.delta); % input gate

        % a_f(t) = W_fx * x(t) + W_fh * h(t-1) + W_fc * c(t-1)
        maf(t,:) = x(t,:) * cell.W_fx' + mh(t-1, :) * cell.W_fh' + mc(t-1, :) .* cell.W_fc';
        mf(t,:) = active_func(maf(t,:), cell.delta); % forget gate

        % a_c(t) = W_ci * x(t) + W_ch * h(t-1)
        mac(t,:) = x(t,:) * cell.W_cx' +  mh(t-1, :) * cell.W_ch';
        % gac(t) = g(a_c(t))
        mgac(t,:) = active_func(mac(t,:), cell.g);
        % c(t) = f(t) * c(t-1) + i(t) * gac(t)
        mc(t,:) = mf(t,:) .* mc(t-1, :) + mi(t,:) .*  mgac(t,:);

        % a_o(t) = W_ox * x(t) + W_oh * h(t-1) + W_oc * c(t)
        mao(t,:) = x(t,:) * cell.W_ox' + mh(t-1, :) * cell.W_oh' + mc(t,:) .* cell.W_oc';
        % o(t) = delta(a_o(t))
        mo(t,:) = active_func(mao(t,:), cell.delta); % output gate
        
        % gc(t) = g(c(t))
        mgc(t, :) = active_func(mc(t,:), cell.g);
        % h(t) = o(t) * gc(t)
        mh(t,:) = mo(t,:) .* mgc(t, :);
    end
%% 
    cell.x = x;
    cell.mi = mi;
    cell.mai = mai;
    cell.mf = mf;
    cell.maf = maf;
    cell.mc = mc;
    cell.mac = mac;
    cell.mgac = mgac;
    cell.mo = mo;
    cell.mao = mao;
    cell.mgc = mgc;
    cell.mh = mh;
end