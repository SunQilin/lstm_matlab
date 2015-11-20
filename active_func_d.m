function y = active_func_d(x, type)
%
%
    switch type
        case 'sigm'
             y = x .* (1 - x);
        case ['linear', 'softmax']
            y = x;
        case 'tanh_opt'
            y = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * x.^2);
        case 'ReLU'
        case 'loglinear'
    end
end