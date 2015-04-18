

clear
clc
seq_len = 200;
len_in = 20;
len_out = 200;
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
% %% dW_ix
% dW_ix = zeros(len_out, len_in + 1);
% for i = 1 : len_out
%     for j = 1 : len_in + 1
%         lstmcell2 = lstmcell;
%         lstmcell2.W_ix(i, j) = lstmcell2.W_ix(i, j) + 1e-2;
%         lstmcell2 = lstmcellff(lstmcell2, x, y);
%         e = y - lstmcell2.mh;
%         loss_2 = sum(sum(e .* e)) / 2 / seq_len;
%         lstmcell = lstmcellbp(lstmcell, -e);
%         dW_ix(i,j) = (loss_2 - loss_1) / 1e-2;
%     end
% end
% aaa = (lstmcell.dW_ix - dW_ix);
% disp(['dW_ix  '  num2str(max(max(abs(aaa) ./abs(dW_ix)))) '  ' num2str(max(max(abs(aaa))))])
% 
% 
% 
% %% dW_fx
% dW_fx = zeros(len_out, len_in + 1);
% for i = 1 : len_out
%     for j = 1 : len_in + 1
%         lstmcell2 = lstmcell;
%         lstmcell2.W_fx(i, j) = lstmcell2.W_fx(i, j) + 1e-2;
%         lstmcell2 = lstmcellff(lstmcell2, x, y);
%         e = y - lstmcell2.mh;
%         loss_2 = sum(sum(e .* e)) / 2 / seq_len;
%         lstmcell = lstmcellbp(lstmcell, -e);
%         dW_fx(i,j) = (loss_2 - loss_1) / 1e-2;
%     end
% end
% aaa = (lstmcell.dW_fx - dW_fx);
% disp(['dW_fx  '  num2str(max(max(abs(aaa) ./abs(dW_fx)))) '  ' num2str(max(max(abs(aaa))))])
% 
% 
% %% dW_cx
% dW_cx = zeros(len_out, len_in + 1);
% for i = 1 : len_out
%     for j = 1 : len_in + 1
%         lstmcell2 = lstmcell;
%         lstmcell2.W_cx(i, j) = lstmcell2.W_cx(i, j) + 1e-2;
%         lstmcell2 = lstmcellff(lstmcell2, x, y);
%         e = y - lstmcell2.mh;
%         loss_2 = sum(sum(e .* e)) / 2 / seq_len;
%         lstmcell = lstmcellbp(lstmcell, -e);
%         dW_cx(i,j) = (loss_2 - loss_1) / 1e-2;
%     end
% end
% aaa = (lstmcell.dW_cx - dW_cx);
% disp(['dW_cx  '  num2str(max(max(abs(aaa) ./abs(dW_cx)))) '  ' num2str(max(max(abs(aaa))))])
% 
% 
% %% dW_ox
% dW_ox = zeros();
% for i = 1 : len_out
%     for j = 1 : len_in + 1
%         lstmcell2 = lstmcell;
%         lstmcell2.W_ox(i, j) = lstmcell2.W_ox(i, j) + 1e-2;
%         lstmcell2 = lstmcellff(lstmcell2, x, y);
%         e = y - lstmcell2.mh;
%         loss_2 = sum(sum(e .* e)) / 2 / seq_len;
%         lstmcell = lstmcellbp(lstmcell, -e);
%         dW_ox(i,j) = (loss_2 - loss_1) / 1e-2;
%     end
% end
% aaa = (lstmcell.dW_ox - dW_ox);
% disp(['dW_ox  '  num2str(max(max(abs(aaa) ./abs(dW_ox)))) '  ' num2str(max(max(abs(aaa))))])
% 
% 
% 
% %% dW_ih
% dW_ih = zeros(len_out, len_out);
% for i = 1 : len_out
%     for j = 1 : len_out
%         lstmcell2 = lstmcell;
%         lstmcell2.W_ih(i, j) = lstmcell2.W_ih(i, j) + 1e-2;
%         lstmcell2 = lstmcellff(lstmcell2, x, y);
%         e = y - lstmcell2.mh;
%         loss_2 = sum(sum(e .* e)) / 2 / seq_len;
%         lstmcell = lstmcellbp(lstmcell, -e);
%         dW_ih(i,j) = (loss_2 - loss_1) / 1e-2;
%     end
% end
% aaa = (lstmcell.dW_ih - dW_ih);
% disp(['dW_ih  '  num2str(max(max(abs(aaa) ./abs(dW_ih)))) '  ' num2str(max(max(abs(aaa))))])
% 
% 
% 
% %% dW_fh
% dW_fh = zeros(len_out, len_out);
% for i = 1 : len_out
%     for j = 1 : len_out
%         lstmcell2 = lstmcell;
%         lstmcell2.W_fh(i, j) = lstmcell2.W_fh(i, j) + 1e-2;
%         lstmcell2 = lstmcellff(lstmcell2, x, y);
%         e = y - lstmcell2.mh;
%         loss_2 = sum(sum(e .* e)) / 2 / seq_len;
%         lstmcell = lstmcellbp(lstmcell, -e);
%         dW_fh(i,j) = (loss_2 - loss_1) / 1e-2;
%     end
% end
% aaa = (lstmcell.dW_fh - dW_fh);
% disp(['dW_fh  '  num2str(max(max(abs(aaa) ./abs(dW_fh)))) '  ' num2str(max(max(abs(aaa))))])
% 
% 
% %% dW_ch
% dW_ch = zeros(len_out, len_out);
% for i = 1 : len_out
%     for j = 1 : len_out
%         lstmcell2 = lstmcell;
%         lstmcell2.W_ch(i, j) = lstmcell2.W_ch(i, j) + 1e-2;
%         lstmcell2 = lstmcellff(lstmcell2, x, y);
%         e = y - lstmcell2.mh;
%         loss_2 = sum(sum(e .* e)) / 2 / seq_len;
%         lstmcell = lstmcellbp(lstmcell, -e);
%         dW_ch(i,j) = (loss_2 - loss_1) / 1e-2;
%     end
% end
% aaa = (lstmcell.dW_ch - dW_ch);
% disp(['dW_ch  '  num2str(max(max(abs(aaa) ./abs(dW_ch)))) '  ' num2str(max(max(abs(aaa))))])
% 
% 
% %% dW_oh
% dW_oh = zeros(len_out, len_out);
% for i = 1 : len_out
%     for j = 1 : len_out
%         lstmcell2 = lstmcell;
%         lstmcell2.W_oh(i, j) = lstmcell2.W_oh(i, j) + 1e-2;
%         lstmcell2 = lstmcellff(lstmcell2, x, y);
%         e = y - lstmcell2.mh;
%         loss_2 = sum(sum(e .* e)) / 2 / seq_len;
%         lstmcell = lstmcellbp(lstmcell, -e);
%         dW_oh(i,j) = (loss_2 - loss_1) / 1e-2;
%     end
% end
% aaa = (lstmcell.dW_oh - dW_oh);
% disp(['dW_oh  '  num2str(max(max(abs(aaa) ./abs(dW_oh)))) '  ' num2str(max(max(abs(aaa))))])
% 
% 
% %% dW_ic
% dW_ic = zeros(len_out, 1);
% for i = 1 : len_out
%     lstmcell2 = lstmcell;
%     lstmcell2.W_ic(i) = lstmcell2.W_ic(i) + 1e-2;
%     lstmcell2 = lstmcellff(lstmcell2, x, y);
%     e = y - lstmcell2.mh;
%     loss_2 = sum(sum(e .* e)) / 2 / seq_len;
%     lstmcell = lstmcellbp(lstmcell, -e);
%     dW_ic(i) = (loss_2 - loss_1) / 1e-2;
% end
% aaa = (lstmcell.dW_ic - dW_ic);
% disp(['dW_ic  '  num2str(max(max(abs(aaa) ./abs(dW_ic)))) '  ' num2str(max(max(abs(aaa))))])
% 
% 
% 
% %% dW_fc
% dW_fc = zeros(len_out, 1);
% for i = 1 : len_out
%     lstmcell2 = lstmcell;
%     lstmcell2.W_fc(i) = lstmcell2.W_fc(i) + 1e-2;
%     lstmcell2 = lstmcellff(lstmcell2, x, y);
%     e = y - lstmcell2.mh;
%     loss_2 = sum(sum(e .* e)) / 2 / seq_len;
%     lstmcell = lstmcellbp(lstmcell, -e);
%     dW_fc(i) = (loss_2 - loss_1) / 1e-2;
% end
% aaa = (lstmcell.dW_fc - dW_fc);
% disp(['dW_fc  '  num2str(max(max(abs(aaa) ./abs(dW_fc)))) '  ' num2str(max(max(abs(aaa))))])
% 
% 
% %% dW_oc
% dW_oc = zeros(len_out, 1);
% for i = 1 : len_out
%     lstmcell2 = lstmcell;
%     lstmcell2.W_oc(i) = lstmcell2.W_oc(i) + 1e-2;
%     lstmcell2 = lstmcellff(lstmcell2, x, y);
%     e = y - lstmcell2.mh;
%     loss_2 = sum(sum(e .* e)) / 2 / seq_len;
%     lstmcell = lstmcellbp(lstmcell, -e);
%     dW_oc(i) = (loss_2 - loss_1) / 1e-2;
% end
% aaa = (lstmcell.dW_oc - dW_oc);
% disp(['dW_oc  '  num2str(max(max(abs(aaa) ./abs(dW_oc)))) '  ' num2str(max(max(abs(aaa))))])









for i = 1:100
    lstmcell = lstmcellff(lstmcell, x, y);
    e = y - lstmcell.mh;
    loss(i) = sum(sum(e .* e)) / 2 / seq_len;
    lstmcell = lstmcellbp(lstmcell, -e);
    lstmcell = lstmcellupdate(lstmcell);

end
