function p = predictOneVsAll(all_theta, X)

m = size(X, 1);

p = zeros(size(X, 1), 1);

X = [ones(m, 1) X];

predicho = sigmoide(X*all_theta');
[predicho_max, p] = max(predicho, [], 2);

end