function [J,grad] = costeLinearRegularizado(X,y,theta,lambda)
%Toma valores de X, y , theta y lambda para devolver el gradiente y el
%coste de los datos.

m = length(y);
%el calcuo del coste 

J = (1/(2*m))*sum((X*theta-y).^2) - (lambda/(2*m))*sum(theta(2:end).^2);

%el factor de regularizacion, que para 1 siempre es 0
reg = (lambda/m).*theta;
reg(1) = 0;


%el calculo del gradiente 
grad = ((1/m) .* X' * (X*theta - y)) + reg;

end

