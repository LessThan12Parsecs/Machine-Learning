%Practica 3 Aprendizaje Automatico y Big Data
%Emanuel Ramirez Catapano
%Regresion logistica multiclase
%Realizado con los datos cargados del archivo ex3data1.mat

%cargamos los datos en X e y
load('ex3data1.mat');
% graficamos 
m =size(X,1);
rand_indices = randperm(m);
sel = X(rand_indices(1:100),:);
displayData(sel);


[all_theta] = oneVsAll(X,y,10,1);
[p] = predictOneVsAll(all_theta,X);
acertados = p==y;
fprintf('\nEl porcentaje de aciertos es de : %f\n', (sum(acertados)/5000)*100);

