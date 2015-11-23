function [J,grad] = lrCostFunction(theta,X,y,lambda)
%  Calculo del coste para la version regularizada.
    m = length(X(:,1));
    prueba = X*theta;
    sigm = sigmoide(prueba);
    %calculamos la funcion de coste
    J =(1/m)...
        *sum(-y.*log(sigm)- (1-y).*log(1-sigm))+...
        (lambda/(2*m))*norm(theta([2:end]))^2;
    %luego el gradiente.
    G = (lambda/m).*theta; G(1) = 0;
    grad = (1/m).*X'*(sigm-y)+G; 
    
    grad = grad(:);
end

