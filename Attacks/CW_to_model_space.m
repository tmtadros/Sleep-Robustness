function [ x, grad ] = CW_to_model_space( x, min_, max_ )

% from (-inf, +inf) to (-1,1)
x = tanh(x);

grad = 1 - x.^2;

% Map from (-1, 1) to (min, max)
a = (min_ + max_) / 2;
b = (max_ - min_) / 2;
x = x * b + a;

grad = grad * b;

end
