function [W] = pesosAleatorios(L_in,L_out)
%Devuelve una matriz W de dimension (L_in,L_out +1) con valores aleatorios.
W = zeros(L_in,L_out + 1);

W = reshape(sin(1:numel(W)),size(W))/10;
end

