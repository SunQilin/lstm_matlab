function cell = lstmcellupdate(cell, learningRate)
%LSTMcELLUPDATE  updates weights and biases with calculated gradients
%   cell = lstmupdate(cell)

if nargin == 2
    cell.learningRate = learningRate;
end
    if(cell.weightPenaltyL2>0)
        
        dW_ix = cell.learningRate * (cell.dW_ix + cell.weightPenaltyL2 * [zeros(size(cell.W_ix,1), 1) cell.W_ix(:,2:end)]);
        dW_ih = cell.learningRate * (cell.dW_ih + cell.weightPenaltyL2 * cell.W_ih);
        dW_ic = cell.learningRate * (cell.dW_ic + cell.weightPenaltyL2 * cell.W_ic);
        
        dW_fx = cell.learningRate * (cell.dW_fx + cell.weightPenaltyL2 * [zeros(size(cell.W_fx,1), 1) cell.W_ix(:,2:end)]);
        dW_fh = cell.learningRate * (cell.dW_fh + cell.weightPenaltyL2 * cell.W_fh);
        dW_fc = cell.learningRate * (cell.dW_fc + cell.weightPenaltyL2 * cell.W_fc);
        
        dW_cx = cell.learningRate * (cell.dW_cx + cell.weightPenaltyL2 * [zeros(size(cell.W_cx,1), 1) cell.W_ix(:,2:end)]);
        dW_ch = cell.learningRate * (cell.dW_ch + cell.weightPenaltyL2 * cell.W_ch);
        
        dW_ox = cell.learningRate * (cell.dW_ox + cell.weightPenaltyL2 * [zeros(size(cell.W_ox,1), 1) cell.W_ix(:,2:end)]);
        dW_oh = cell.learningRate * (cell.dW_oh + cell.weightPenaltyL2 * cell.W_oh);
        dW_oc = cell.learningRate * (cell.dW_oc + cell.weightPenaltyL2 * cell.W_oc);
        
    else
        dW_ix = cell.learningRate * cell.dW_ix;
        dW_ih = cell.learningRate * cell.dW_ih;
        dW_ic = cell.learningRate * cell.dW_ic;
%         db_i = cell.learningRate * cell.db_i;
        
        dW_fx = cell.learningRate * cell.dW_fx;
        dW_fh = cell.learningRate * cell.dW_fh;
        dW_fc = cell.learningRate * cell.dW_fc;
%         db_f = cell.learningRate * cell.db_f;
        
        dW_cx = cell.learningRate * cell.dW_cx;
        dW_ch = cell.learningRate * cell.dW_ch;
%         db_c = cell.learningRate * cell.db_c;
        
        dW_ox = cell.learningRate * cell.dW_ox;
        dW_oh = cell.learningRate * cell.dW_oh;
        dW_oc = cell.learningRate * cell.dW_oc;
%         db_o = cell.learningRate * cell.db_o;
    end
    
    if(cell.momentum>0)
        cell.vW_ix = cell.momentum * cell.vW_ix - dW_ix;
        dW_ix = cell.vW_ix;
        cell.vW_ih = cell.momentum * cell.vW_ih - dW_ih;
        dW_ih = cell.vW_ih;
        cell.vW_ic = cell.momentum * cell.vW_ic - dW_ic;
        dW_ic = cell.vW_ic;
%         cell.vb_i = cell.momentum * cell.vb_i - db_i;
%         db_i = cell.vb_i;

        cell.vW_fx = cell.momentum * cell.vW_fx - dW_fx;
        dW_fx = cell.vW_fx;
        cell.vW_fh = cell.momentum * cell.vW_fh - dW_fh;
        dW_fh = cell.vW_fh;
        cell.vW_fc = cell.momentum * cell.vW_fc - dW_fc;
        dW_fc = cell.vW_fc;
%         cell.vb_f = cell.momentum * cell.vb_f - db_f;
%         db_f = cell.vb_f;

        cell.vW_cx =  cell.momentum * cell.vW_cx - dW_cx;
        dW_cx = cell.vW_cx;
        cell.vW_ch = cell.momentum * cell.vW_ch - dW_ch;
        dW_ch = cell.vW_ch;
%         cell.vb_c = cell.momentum * cell.vb_c - db_c;
%         db_c = cell.vb_c;

        cell.vW_ox = cell.momentum * cell.vW_ox - dW_ox;
        dW_ox = cell.vW_ox;
        cell.vW_oh = cell.momentum * cell.vW_oh - dW_oh;
        dW_oh = cell.vW_oh;
        cell.vW_oc = cell.momentum * cell.vW_oc - dW_oc;
        dW_oc = cell.vW_oc;
%         cell.vb_o = cell.momentum * cell.vb_o - db_o;
%         db_o = cell.vb_o;
    end
        
    cell.W_ix = cell.W_ix + dW_ix;
    cell.W_ih = cell.W_ih + dW_ih;
    cell.W_ic = cell.W_ic + dW_ic;
%     cell.b_i = cell.b_i + db_i;
    
    cell.W_fx = cell.W_fx + dW_fx;
    cell.W_fh = cell.W_fh + dW_fh;
    cell.W_fc = cell.W_fc + dW_fc;
%     cell.b_f = cell.b_f + db_f;
    
    cell.W_cx = cell.W_cx + dW_cx;
    cell.W_ch = cell.W_ch + dW_ch;
%     cell.b_c = cell.b_c + db_c;
    
    cell.W_ox = cell.W_ox + dW_ox;
    cell.W_oh = cell.W_oh + dW_oh;
    cell.W_oc = cell.W_oc + dW_oc;
%     cell.b_o = cell.b_o + db_o;
   
end