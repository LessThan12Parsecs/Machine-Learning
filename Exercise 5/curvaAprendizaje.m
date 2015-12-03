function [error_entre, error_valid] = ...
    curvaAprendizaje(X, y, Xval, yval, lambda)

%tama?o de los ejemplos de entrenamiento 
m = size(X, 1);
%error del conjunto de entrenamiento y de validacion.
error_entre = zeros(m, 1);
error_valid   = zeros(m, 1);

for i = 1:m
    X_sub = X(1:i, :);
    y_sub = y(1:i); 

    theta = entrenarRegresion(X_sub, y_sub, lambda);

    error_entre(i) = costeLinearRegularizado(X_sub, y_sub, theta, 0);
    error_valid(i) = costeLinearRegularizado(Xval, yval, theta, 0);
end
end