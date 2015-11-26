function [W] = pesosAleatorios(L_in,L_out)
%Devuelve una matriz W de dimension (L_in,L_out +1) con valores aleatorios.
epsilon = 0.12;
W = (rand(L_in,L_out + 1)*2*epsilon - epsilon);
end

