function [porcentaje] = evaluacionRegLog(theta,X,y)
    %Evalua los resultados de la regresion logistica comprobando en los
    %ejemplos de entrenamiento cuantos se han clasificado correctamente. 
    prob = sigmoide(X*theta);
    yes = prob >= 0.5; %valores predecidos que estan aprobados.
    correct = yes == y; %valores que concuerdan con los datos reales.
    porcentaje = (sum(correct)/length(y))*100; %porcentaje entre ambos.
end

