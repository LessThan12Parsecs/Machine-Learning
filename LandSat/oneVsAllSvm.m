function [all_models] = oneVsAllSvm(X,y,num_etiquetas)
%ONEVSALL entrena varios clasificadores por regresi?n log?stica y devuelve 
%el resultado en una matriz all_theta , donde la fila i ?sima
%corresponde al clasificador de la etiqueta i ?sima

%contamos el numero de ejemplos de entramiento 
m = size(X,1);
%guardamos el numero de atributos de los ejemplos
n = size(X,2);

%agregamos una columna de 1 a X

X = [ones(m,1) X];

%inicializamos todos los valores de todos los thetas
all_thetas = zeros(num_etiquetas, n+1);

%valores iniciales para theta 
theta_inicio = zeros(n+1,1);
%guardamos las opciones que queremos para el fmincg
options = optimset('GradObj', 'on', 'MaxIter', 50);

%calculamos los valores para cada etiqueta

for i = 1:num_etiquetas
   c = i * ones(size(y));
   actual = y==c;
   [model] = svmTrain(X,actual,valores(iC),@(x1,x2) gaussianKernel(x1,x2,valores(iSigma)));
   all_models(i,:) = theta; 
end

end

