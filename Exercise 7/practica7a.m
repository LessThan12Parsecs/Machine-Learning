%Practica 7a Aprendizaje Automatico y Big Data
%Emanuel Ramirez Catapano
%Clustering, K-Means
%Realizado con los datos cargados de ex7data2.mat


%cargamos X de ex7data2.mat
load('ex7data2.mat');
%para los valores de prueba de centroide usamos los siguientes.
centroids = [3 3; 6 2; 8 5];
runkMeans(X,centroids,10,true);
K = 5;
%ahora con valores aleatorios para los centroides realizamos la llama a
%runkMeans de nuevo.
randidx = randperm(size(X,1));
centroides = X(randidx(1:K),:);
runkMeans(X,centroides,10,true);

