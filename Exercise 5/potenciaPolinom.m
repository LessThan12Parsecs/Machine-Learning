function [X_polinom] = potenciaPolinom(X, p)
X_polinom = zeros(numel(X), p);

%tama?o de los ejemplos de entrenamiento.
m = size(X,1);

for i=1:m

    polinomio = zeros(p, 1);

    for j=1:p
        polinomio(j) =  X(i).^j;
    end

    X_polinom(i, :) = polinomio;
end


end