function [J grad] = costeRN(params_rn,num_entradas,num_ocultas,...
    num_etiquetas,X,y,lambda)
    %costeRN calcula el coste y el gradiente de una red neuronal de dos
    %capas
    
    Theta1 = reshape(params_rn(1:num_ocultas*(num_entradas+1)),... 
      num_ocultas,(num_entradas+1));
    Theta2 = reshape(params_rn((1+(num_ocultas*(num_entradas+1))):end),... 
      num_etiquetas,(num_ocultas+1));
   
  %el numero de ejemplos de entrenamiento
  m = size(X,1);
  
  J = 0;
  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));
  
 
 yAux = eye(num_etiquetas);
 y = yAux(y,:);
  
   
   
end

