function [Y] = sigmoide(X)
%metodo que aplica la funcion sigmoide a los elementos que entren en X
%param X: datos de entrada
%out Y: X luego de ser aplicada la funcion sigmoide.
   Y = 1./(1 + exp(-X));
end

