% This is a toy example of the use of the code, and the network created has
% only a lstm layer
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
%%

%% train the network
for i = 1:100
    lstmcell = lstmcellff(lstmcell, x, y);
    e = y - lstmcell.mh;
    loss(i) = sum(sum(e .* e)) / 2 / seq_len;
    lstmcell = lstmcellbp(lstmcell, -e);
    lstmcell = lstmcellupdate(lstmcell);
end
plot(loss);