function [p] = predecirRedNeu(Theta1,Theta2,X)

m = size(X, 1);
p = zeros(size(X, 1), 1);

% Input Layer
a1 = [ones(m, 1), X];
z2 = Theta1 * a1';

% Hidden Layer
a2 = sigmoide(z2);  
a2 = [ones(1,size(a2,2)); a2];

% Output layer
z3 = Theta2 * a2;
a3 = sigmoide(z3); 
%Calculo del valor maximo.
[val, p] = max(a3', [], 2);

end

