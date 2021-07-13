function [ x ] = CW_to_attack_space( x, min_, max_ )
% map from [min_, max_] to [-1, +1]

a = (min_ + max_) / 2;
b = (max_ - min_) / 2;
x = (x-a)/b;

x = x * 0.9999999;
x = atanh(x);
end

