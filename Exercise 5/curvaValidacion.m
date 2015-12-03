function [lambda_vec, error_entre, error_valid] = ...
    curvaValidacion(X, y, Xval, yval)

lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

error_entre = zeros(length(lambda_vec), 1);
error_valid = zeros(length(lambda_vec), 1);
for i = 1:length(lambda_vec)
	lambda = lambda_vec(i);
	[theta] = entrenarRegresion(X, y, lambda);
	error_entre(i) = costeLinearRegularizado(X, y, theta, 0);
	error_valid(i) = costeLinearRegularizado(Xval, yval, theta, 0);
end
end