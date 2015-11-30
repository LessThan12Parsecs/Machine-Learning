%Practica 5 - Aprendizaje Automatico y Big Data
%Emanuel Ram?rez Catapano
%Regularizacion Linear con sesgo y varianza
%Realizado cargando los datos de ex5data1.mat

%Cargamos los datos
load('ex5data1.mat');

m = size(X,1);
theta = [1;1];
lambda = 1;

[J,grad] = costeLinearRegularizado([ones(m,1) X],y,theta,lambda);

lambda = 0;
[theta] = entrenarRegresion([ones(m, 1) X], y, lambda);

%  Plot fit over the data
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
hold on;
plot(X, [ones(m, 1) X]*theta, '--', 'LineWidth', 2)
hold off;
