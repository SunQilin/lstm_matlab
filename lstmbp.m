function net = lstmbp(net)
% LSTMBP
%
%
    n = numel(net.layers);
    % partial derivative for the output layer
    switch net.layers{n}.active_func
        case 'sigm'
            d{n} = - net.e .* (net.layers{n}.a .* (1 - net.layers{n}.a));
        case {'softmax','linear'}
            d{n} = - net.e;
    end
    d_next = d{n} * net.layers{n}.W;
    % compute the partial derivative for the hidden layers layer-by-layer 
    for i = (n - 1) : -1 : 2
        switch net.layers{i}.type
            case 'lstm'
                net.layers{i}.cell = lstmcellbp(net.layers{i}.cell, d_next(:, 2:end));
                d_next = net.layers{i}.cell.dx;
            case 'blstm'
                cell_len = 1/2 * (size(d_next, 2) - 1);     % split the partial derivative into two part
                net.layers{i}.cellf = lstmcellbp(net.layers{i}.cellf, d_next(:, 2 : cell_len + 1)); % the forword part
                net.layers{i}.cellb = lstmcellbp(net.layers{i}.cellb, flipud(d_next(:, cell_len + 2 : end)));   % the backword part
                d_next = (net.layers{i}.cellf.dx + flipud(net.layers{i}.cellb.dx)); % sumaation of the two part
                
            case 'normal'
                d_act = active_func_d(net.layers{i}.a, net.layers{i}.active_func);
%                 switch net.layers{i}.active_func
%                     case 'sigm'
%                         d_act = net.a{i} .* (1 - net.a{i});
%                     case 'tanh_opt'
%                         d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * net.a{i}.^2);
%                 end
                
                % Backpropagate first derivatives % Bishop (5.56)
                d{i} = d_next .* d_act;
                d_next = d{i}(:, 2:end) * net.layers{i}.W;             
        end
    
    end
    
    for i = 1 : n
        if  strcmp(net.layers{i}.type, 'normal')
            if i == n
                net.layers{i}.dW = (d{i}' * net.layers{i - 1}.a) / size(d{i}, 1);
            else
                net.layers{i}.dW = (d{i}(:, 2:end)' * net.layers{i - 1}.a) / size(d{i}, 1);
            end
        end
    end
end