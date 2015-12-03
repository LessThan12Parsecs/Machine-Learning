%Practica 5 - Aprendizaje Automatico y Big Data
%Emanuel Ramirez Catapano
%Regularizacion Linear con sesgo y varianza
%Realizado cargando los datos de ex5data1.mat

load ('ex5data1.mat');
m = size(X, 1);

% Realizamos el calculo del coste  y gradiente con theta [1;1]
theta = [1 ; 1];
J = costeLinearRegularizado([ones(m, 1) X], y, theta, 1);

fprintf(['El coste en theta = [1 ; 1]: %f '], J);

fprintf('\n Presione enter para continuar.\n');
pause;

theta = [1 ; 1];
[J, grad] = costeLinearRegularizado([ones(m, 1) X], y, theta, 1);

fprintf(['El gradiente en  = [1 ; 1]:  [%f; %f] '],...
         grad(1), grad(2));

     
fprintf('\n Presione enter para continuar.\n');
pause;

%  Entrenamos la regresion lineal para lambda 0
lambda = 0;
[theta] = entrenarRegresion([ones(m, 1) X], y, lambda);

% Lo graficamos.
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Cambio en el nivel del agua (x)');
ylabel('Flujo de agua saliendo por la presa(y)');
hold on;
plot(X, [ones(m, 1) X]*theta, '--', 'LineWidth', 2)
hold off;

fprintf('\n Presione enter para continuar.\n');
pause;

%Curvas de Aprendizaje.===================================================
lambda = 0;
[error_entre, error_valid] = curvaAprendizaje([ones(m, 1) X], y, ...
[ones(size(Xval, 1), 1) Xval], yval,lambda);

plot(1:m, error_entre, 1:m, error_valid);
title('Curvas de Aprendizaje para regresion lineal')
legend('Entrenamiento', 'Validacion')
xlabel('Numero de ejemplos de entrenamiento')
ylabel('Error')
axis([0 13 0 150])

fprintf('# Ejemplos de Entrenamiento\tError de Entrenamiento\tError de Validacion\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_entre(i), error_valid(i));
end

fprintf('\n Presione enter para continuar.\n');
pause;

%Mapeo en regresion polinomial.====================================
p = 8;

% Mapeo X en potenciaPolino y normalizo
X_polinom = potenciaPolinom(X, p);
[X_polinom, mu, sigma] = featureNormalize(X_polinom);  % Normalizar
X_polinom = [ones(m, 1), X_polinom];                   % A?adir unos

% Mapeo X_poly_test y normalizo con mu y sigma
X_poly_test = potenciaPolinom(Xtest, p);
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];  % A?adir unos

% Mapo X_poly_val y normalizo con mu y sigma
X_poly_val = potenciaPolinom(Xval, p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];   % A?adir unos

fprintf('Ejemplo de entrenamiento normalizado 1:\n');
fprintf('  %f  \n', X_polinom(1, :));

fprintf('\n Presione enter para continuar.\n');
pause;

%Ajuste de Regresion polinomial ===========================================
lambda = 0;
[theta] = entrenarRegresion(X_polinom, y, lambda);

% Plot training data and fit
figure(1);
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
plotFit(min(X), max(X), mu, sigma, theta, p);
xlabel('Cambio en el nivel del agua (x)');
ylabel('Flujo de agua que sale por la presa(y)');
title (sprintf('Ajuste de regresion polinomial(lambda = %f)', lambda));

figure(2);
[error_train, error_val] = ...
    curvaAprendizaje(X_polinom, y, X_poly_val, yval, lambda);
plot(1:m, error_train, 1:m, error_val);

title(sprintf('Curva de aprendizaje de regresion polinomial (lambda = %f)', lambda));
xlabel('Numero de ejemplos de entrenamiento')
ylabel('Error')
axis([0 13 0 100])
legend('Entrenamiento', 'Validacion')

fprintf('Regresion polinomial (lambda = %f)\n\n', lambda);
fprintf('#Ejemplos entrenamiento\tError entrena\tError Validacion\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

fprintf('\n Presione enter para continuar.\n');
pause;


%Eleccion de lambda========================================================

[lambda_vec, error_entre, error_valid] = ...
    curvaValidacion(X_polinom, y, X_poly_val, yval);


plot(lambda_vec, error_entre, lambda_vec, error_valid);
legend('Entrenamiento', 'Validacion');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tError entrenamiento\terror Validacion\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_entre(i), error_valid(i));
end

fprintf('\n Presione enter para continuar.\n');
