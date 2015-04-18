function lstmcell = lstmcellsetup(inputlen, outputlen, opt, active)
%lstmcell create a lstmcell layer for a Feedforword Backpropagate Neural Network and it can be used as a active function conveniently
%   lstmcell = lstmcellsetup(inputlen, outputlen) 

%% lstmcell setup
    lstmcell.inputlen = inputlen;
    lstmcell.outputlen = outputlen;
    lstmcell.delta = active{1};
    lstmcell.g = active{2};
    %lstmcell.h = 'tanh';
    lstmcell.learningRate = opt.learningRate;
    lstmcell.momentum = opt.momentum;
    lstmcell.weightPenaltyL2 = opt.weightPenaltyL2;
    lstmcell.scaling_learningRate = opt.scaling_learningRate;

    
    %i_t
    lstmcell.W_ix = (rand(outputlen, inputlen + 1) - 0.5) / inputlen;
    lstmcell.W_ih = (rand(outputlen, outputlen) - 0.5) / outputlen;
    lstmcell.W_ic = (rand(outputlen, 1) - 0.5) / outputlen; % diagonal matrix represent by vertor
    
    %f_t
    lstmcell.W_fx = (rand(outputlen, inputlen + 1) - 0.5) / inputlen;
    lstmcell.W_fh = (rand(outputlen, outputlen) - 0.5) / outputlen;
    lstmcell.W_fc = (rand(outputlen, 1) - 0.5) / outputlen; % diagonal matrix represent by vertor
    
    %c_t
    lstmcell.W_cx = (rand(outputlen, inputlen + 1) - 0.5) / inputlen;
    lstmcell.W_ch = (rand(outputlen, outputlen) - 0.5) / outputlen;
    
    %o_t
    lstmcell.W_ox = (rand(outputlen, inputlen + 1) - 0.5) / inputlen;
    lstmcell.W_oh = (rand(outputlen, outputlen) - 0.5) / outputlen;
    lstmcell.W_oc = (rand(outputlen, 1) - 0.5) / outputlen; % diagonal matrix represent by vertor
    
    if lstmcell.momentum  > 0
        lstmcell.vW_ix = zeros(size(lstmcell.W_ix));
        lstmcell.vW_ih = zeros(size(lstmcell.W_ih));
        lstmcell.vW_ic = zeros(size(lstmcell.W_ic));

        lstmcell.vW_fx = zeros(size(lstmcell.W_fx));
        lstmcell.vW_fh = zeros(size(lstmcell.W_fh));
        lstmcell.vW_fc = zeros(size(lstmcell.W_fc));


        lstmcell.vW_cx = zeros(size(lstmcell.W_cx));
        lstmcell.vW_ch = zeros(size(lstmcell.W_ch));


        lstmcell.vW_ox = zeros(size(lstmcell.W_ox));
        lstmcell.vW_oh = zeros(size(lstmcell.W_oh));
        lstmcell.vW_oc = zeros(size(lstmcell.W_oc ));
    end
    
end