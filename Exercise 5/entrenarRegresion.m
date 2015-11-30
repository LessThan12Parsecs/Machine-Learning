function [theta] = entrenarRegresion(X,y,lambda)

% Inicializamos theta
initial_theta = zeros(size(X, 2), 1); 

% Creamos funcion de coste
funcionCoste = @(t) costeLinearRegularizado(X, y, t, lambda);
options = optimset('MaxIter', 200, 'GradObj', 'on');

% Utilizamos el fmincg para minimizar el theta
theta = fmincg(funcionCoste, initial_theta, options);

end

