function [J,grad] = coste(theta,X,y)
%   Recibe valores de theta, un vector con datos X y uno con
%   resultados y, y devuelve el valor de la funcion coste en J y un vector
%   con los valores del gradiente grad.
    m = length(X(:,1));
    sigm = sigmoide(X*theta);
    %calculamos la funcion de coste
    J = (1/m)*sum(-y.*log(sigm)-(1-y).*log(1-sigm));
    %luego el gradiente.
    grad = (1/m).*X'*(sigm-y);     
end