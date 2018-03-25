function [ W ] = W_calc( k, idx, sumd )
% Calculates the (normalized?) within-cluster dispersion
% Also described as pooled within-cluster sum of distances around the cluster means

% maybe include some normalizing measures here so we can compare them?

% number of points per cluster:
nums = zeros(k,1);
for i = 1:k
    nums(i) = sum(idx(idx == i));
end

W = sum(0.5*sumd./nums);

end

