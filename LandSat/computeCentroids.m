function centroids = computeCentroids(X,idx,K)
%X son los ejemplos de entrenamiento
%idx son los indicies de los centroides mas cercanos a los ejemplos
%de entrenamiento.
%calcula la nueva posicion del centroide con la media de los ejemplos
%asignados a el

[m n] = size(X);
centroids = zeros(K, n);
for k = 1:K
    D = idx == k;
	centroids(k,:) = mean(X(D,:));
end
end



