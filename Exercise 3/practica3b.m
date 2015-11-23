%Practica 3 Aprendizaje Automatico y Big Data
%Emanuel Ramirez Catapano
%Prediccion con redes neuronales.
%Realizado con los datos cargados del archivo ex3data1.mat y ex3weights.mat

load('ex3data1.mat');
m =size(X,1);
load('ex3weights.mat');

prediccion = predecirRedNeu(Theta1,Theta2,X);
fprintf('\n El porcentaje de aciertos con la red es: %f \n ',(sum(prediccion==y)/5000)*100);



rp = randperm(m);

for i = 1:m
    % Display 
    fprintf('\nMostrando imagen de ejemplo\n');
    displayData(X(rp(i), :));

    pred = predecirRedNeu(Theta1, Theta2, X(rp(i),:));
    fprintf('\nPrediccion de red neuronal: %d (digit %d)\n', pred, mod(pred, 10));
    
    % Pause
    fprintf('Presione enter para continuar\n');
    pause;
end