function [idx] = findClosestCentroids(X,centroids)
%X es una matriz M*N con los ejemplos de entramiento.
%centroids es una matriz k*n con las coordenadas de los centroides.
%Encuentra los centroides mas cercanos a cada ejemplo i.

K = size(centroids,1);
m = size(X,1);
idx = zeros(m,1);
for i=1:m
    distancias = zeros(1,K);
    for j=1:K
        distancias(1,j) = sqrt(sum(power((X(i,:)-centroids(j,:)),2)));
    end
    [d, d_idx] = min(distancias);
    idx(i,1) = d_idx;
end
end

