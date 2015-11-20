function y = active_func(x, type)
switch type
    case 'sigm'
        y = 1./(1+exp(-x));
    case 'tanh_opt'
        y = 1.7159 * tanh(2/3.*x);
    case 'linear'
        y = x;
    case 'softmax'
        x = exp(bsxfun(@minus, x, max(x,[],2)));
        y = bsxfun(@rdivide, x, sum(x, 2));
        
    case 'ReLU'
    case 'loglinear'
end
end