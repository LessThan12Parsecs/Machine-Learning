function [p] = prediccion(theta1,theta2,X)

m = size(X, 1);
p = zeros(size(X, 1), 1);

% Input Layer
a1 = [ones(m, 1), X]; % results in [5000, 401]

z2 = Theta1 * a1'; % results in [25, 5000]

% Hidden Layer
a2 = sigmoide(z2);  % results in [25, 5000]

a2 = [ones(1,size(a2,2)); a2]; % results in [26, 5000]

% Output layer
z3 = Theta2 * a2; % results in [10, 5000]

a3 = sigmoide(z3); % results in [10, 5000]

% calculating max on the transpose of a3 so the index 
% result, p, has the expected dimensions [5000, 1]
[val, p] = max(a3', [], 2);


end

