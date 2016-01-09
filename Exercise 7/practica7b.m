%Practica 7a Aprendizaje Automatico y Big Data
%Emanuel Ramirez Catapano
%Clustering, K-Means
%Compresion de imagenes usando K-Means.

%cargamos la imagen en la matriz A 
A = double(imread('bird_small.png'));
%dividimos entre 255 para tener cada pixel en los valores entre 0 y 1
A = A / 255;
imagesc(A);
%tomamos el tamano de A 
tam_A = size(A);

X = reshape(A,tam_A(1)*tam_A(2),3);

K = 16;
iteraciones = 10;
randidx = randperm(size(X,1));
centroides = X(randidx(1:K),:);
[centroids, idx] = runkMeans(X, centroides, iteraciones);
fprintf('Presione enter para continuar.\n');
pause;



%buscamos los miembros del cluster mas cercano
idx = findClosestCentroids(X, centroids);

%recuperamos la imagen. 
X_nueva = centroids(idx,:);

% hacemos un reshape 
X_nueva = reshape(X_nueva, tam_A(1), tam_A(2), 3);

%La imagen original
subplot(1, 2, 1);
imagesc(A); 
title('Original');

% Hacer un subplot de la comprimida
subplot(1, 2, 2);
imagesc(X_nueva)
title(sprintf('Comprimida, con %d colores.', K));


fprintf('Presione enter para continuar.\n');
pause;
