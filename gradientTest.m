% This file is used to test wheather the gradients compute by lstmcellbp.m is right. 
% This is verified by compare gradients compute by lstmcellbp.m with that compute by numeric difference method.
% If the gradients compute by lstmcellbp.m is right, the relative difference must be small
clear
clc
%% set parameters for the lstmcell
seq_len = 200;
len_in = 10;
len_out = 30;
active_funcs = {'sigm', 'sigm'};
opt.learningRate = 0.1;
opt.weightPenaltyL2 = 0.001;
opt.momentum = 0.5;
opt.scaling_learningRate = 0.5;
lstmcell = lstmcellsetup(len_in, len_out, opt, active_funcs);

x = rand(seq_len, len_in + 1);
y = rand(seq_len, len_out);

lstmcell = lstmcellff(lstmcell, x, y);
e = y - lstmcell.mh;
loss_1 = sum(sum(e .* e)) / 2 / seq_len;
lstmcell = lstmcellbp(lstmcell, e);
%% dW_ix
dW_ix = zeros(len_out, len_in + 1);
for i = 1 : len_out
    for j = 1 : len_in + 1
        lstmcell2 = lstmcell;
        lstmcell2.W_ix(i, j) = lstmcell2.W_ix(i, j) + 1e-2;
        lstmcell2 = lstmcellff(lstmcell2, x, y);
        e = y - lstmcell2.mh;
        loss_2 = sum(sum(e .* e)) / 2 / seq_len;
        lstmcell = lstmcellbp(lstmcell, -e);
        dW_ix(i,j) = (loss_2 - loss_1) / 1e-2;
    end
end
differmatrix = (lstmcell.dW_ix - dW_ix);
disp('')
disp(['dW_ix  max relative difference  '  num2str(max(max(abs(differmatrix) ./abs(dW_ix)))) '  max absolute difference' num2str(max(max(abs(differmatrix))))])



%% dW_fx
dW_fx = zeros(len_out, len_in + 1);
for i = 1 : len_out
    for j = 1 : len_in + 1
        lstmcell2 = lstmcell;
        lstmcell2.W_fx(i, j) = lstmcell2.W_fx(i, j) + 1e-2;
        lstmcell2 = lstmcellff(lstmcell2, x, y);
        e = y - lstmcell2.mh;
        loss_2 = sum(sum(e .* e)) / 2 / seq_len;
        lstmcell = lstmcellbp(lstmcell, -e);
        dW_fx(i,j) = (loss_2 - loss_1) / 1e-2;
    end
end
differmatrix = (lstmcell.dW_fx - dW_fx);
disp(['dW_fx  max relative difference  '  num2str(max(max(abs(differmatrix) ./abs(dW_fx)))) '  max absolute difference' num2str(max(max(abs(differmatrix))))])


%% dW_cx
dW_cx = zeros(len_out, len_in + 1);
for i = 1 : len_out
    for j = 1 : len_in + 1
        lstmcell2 = lstmcell;
        lstmcell2.W_cx(i, j) = lstmcell2.W_cx(i, j) + 1e-2;
        lstmcell2 = lstmcellff(lstmcell2, x, y);
        e = y - lstmcell2.mh;
        loss_2 = sum(sum(e .* e)) / 2 / seq_len;
        lstmcell = lstmcellbp(lstmcell, -e);
        dW_cx(i,j) = (loss_2 - loss_1) / 1e-2;
    end
end
differmatrix = (lstmcell.dW_cx - dW_cx);
disp(['dW_cx  max relative difference  '  num2str(max(max(abs(differmatrix) ./abs(dW_cx)))) '  max absolute difference' num2str(max(max(abs(differmatrix))))])


%% dW_ox
dW_ox = zeros();
for i = 1 : len_out
    for j = 1 : len_in + 1
        lstmcell2 = lstmcell;
        lstmcell2.W_ox(i, j) = lstmcell2.W_ox(i, j) + 1e-2;
        lstmcell2 = lstmcellff(lstmcell2, x, y);
        e = y - lstmcell2.mh;
        loss_2 = sum(sum(e .* e)) / 2 / seq_len;
        lstmcell = lstmcellbp(lstmcell, -e);
        dW_ox(i,j) = (loss_2 - loss_1) / 1e-2;
    end
end
differmatrix = (lstmcell.dW_ox - dW_ox);
disp(['dW_ox  max relative difference  '  num2str(max(max(abs(differmatrix) ./abs(dW_ox)))) '  max absolute difference' num2str(max(max(abs(differmatrix))))])



%% dW_ih
dW_ih = zeros(len_out, len_out);
for i = 1 : len_out
    for j = 1 : len_out
        lstmcell2 = lstmcell;
        lstmcell2.W_ih(i, j) = lstmcell2.W_ih(i, j) + 1e-2;
        lstmcell2 = lstmcellff(lstmcell2, x, y);
        e = y - lstmcell2.mh;
        loss_2 = sum(sum(e .* e)) / 2 / seq_len;
        lstmcell = lstmcellbp(lstmcell, -e);
        dW_ih(i,j) = (loss_2 - loss_1) / 1e-2;
    end
end
differmatrix = (lstmcell.dW_ih - dW_ih);
disp(['dW_ih  max relative difference  '  num2str(max(max(abs(differmatrix) ./abs(dW_ih)))) '  max absolute difference' num2str(max(max(abs(differmatrix))))])



%% dW_fh
dW_fh = zeros(len_out, len_out);
for i = 1 : len_out
    for j = 1 : len_out
        lstmcell2 = lstmcell;
        lstmcell2.W_fh(i, j) = lstmcell2.W_fh(i, j) + 1e-2;
        lstmcell2 = lstmcellff(lstmcell2, x, y);
        e = y - lstmcell2.mh;
        loss_2 = sum(sum(e .* e)) / 2 / seq_len;
        lstmcell = lstmcellbp(lstmcell, -e);
        dW_fh(i,j) = (loss_2 - loss_1) / 1e-2;
    end
end
differmatrix = (lstmcell.dW_fh - dW_fh);
disp(['dW_fh  max relative difference  '  num2str(max(max(abs(differmatrix) ./abs(dW_fh)))) '  max absolute difference' num2str(max(max(abs(differmatrix))))])


%% dW_ch
dW_ch = zeros(len_out, len_out);
for i = 1 : len_out
    for j = 1 : len_out
        lstmcell2 = lstmcell;
        lstmcell2.W_ch(i, j) = lstmcell2.W_ch(i, j) + 1e-2;
        lstmcell2 = lstmcellff(lstmcell2, x, y);
        e = y - lstmcell2.mh;
        loss_2 = sum(sum(e .* e)) / 2 / seq_len;
        lstmcell = lstmcellbp(lstmcell, -e);
        dW_ch(i,j) = (loss_2 - loss_1) / 1e-2;
    end
end
differmatrix = (lstmcell.dW_ch - dW_ch);
disp(['dW_ch  max relative difference  '  num2str(max(max(abs(differmatrix) ./abs(dW_ch)))) '  max absolute difference' num2str(max(max(abs(differmatrix))))])


%% dW_oh
dW_oh = zeros(len_out, len_out);
for i = 1 : len_out
    for j = 1 : len_out
        lstmcell2 = lstmcell;
        lstmcell2.W_oh(i, j) = lstmcell2.W_oh(i, j) + 1e-2;
        lstmcell2 = lstmcellff(lstmcell2, x, y);
        e = y - lstmcell2.mh;
        loss_2 = sum(sum(e .* e)) / 2 / seq_len;
        lstmcell = lstmcellbp(lstmcell, -e);
        dW_oh(i,j) = (loss_2 - loss_1) / 1e-2;
    end
end
differmatrix = (lstmcell.dW_oh - dW_oh);
disp(['dW_oh  max relative difference  '  num2str(max(max(abs(differmatrix) ./abs(dW_oh)))) '  max absolute difference' num2str(max(max(abs(differmatrix))))])


%% dW_ic
dW_ic = zeros(len_out, 1);
for i = 1 : len_out
    lstmcell2 = lstmcell;
    lstmcell2.W_ic(i) = lstmcell2.W_ic(i) + 1e-2;
    lstmcell2 = lstmcellff(lstmcell2, x, y);
    e = y - lstmcell2.mh;
    loss_2 = sum(sum(e .* e)) / 2 / seq_len;
    lstmcell = lstmcellbp(lstmcell, -e);
    dW_ic(i) = (loss_2 - loss_1) / 1e-2;
end
differmatrix = (lstmcell.dW_ic - dW_ic);
disp(['dW_ic  max relative difference  '  num2str(max(max(abs(differmatrix) ./abs(dW_ic)))) '  max absolute difference' num2str(max(max(abs(differmatrix))))])



%% dW_fc
dW_fc = zeros(len_out, 1);
for i = 1 : len_out
    lstmcell2 = lstmcell;
    lstmcell2.W_fc(i) = lstmcell2.W_fc(i) + 1e-2;
    lstmcell2 = lstmcellff(lstmcell2, x, y);
    e = y - lstmcell2.mh;
    loss_2 = sum(sum(e .* e)) / 2 / seq_len;
    lstmcell = lstmcellbp(lstmcell, -e);
    dW_fc(i) = (loss_2 - loss_1) / 1e-2;
end
differmatrix = (lstmcell.dW_fc - dW_fc);
disp(['dW_fc  max relative difference  '  num2str(max(max(abs(differmatrix) ./abs(dW_fc)))) ' max absolute difference' num2str(max(max(abs(differmatrix))))])


%% dW_oc
dW_oc = zeros(len_out, 1);
for i = 1 : len_out
    lstmcell2 = lstmcell;
    lstmcell2.W_oc(i) = lstmcell2.W_oc(i) + 1e-2;
    lstmcell2 = lstmcellff(lstmcell2, x, y);
    e = y - lstmcell2.mh;
    loss_2 = sum(sum(e .* e)) / 2 / seq_len;
    lstmcell = lstmcellbp(lstmcell, -e);
    dW_oc(i) = (loss_2 - loss_1) / 1e-2;
end
differmatrix = (lstmcell.dW_oc - dW_oc);
disp(['dW_oc  max relative difference  '  num2str(max(max(abs(differmatrix) ./abs(dW_oc)))) ' max absolute difference' num2str(max(max(abs(differmatrix))))])

