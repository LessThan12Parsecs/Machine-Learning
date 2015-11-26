%Practica 4 Aprendizaje Automatico y Big Data
%Emanuel Ramirez Catapano
%Entrenamiento de Redes Neuronales
%Utilizando los datos cargados de ex4data1.mat


load('ex4data1.mat');
load('ex4weights.mat');
m = size(X, 1);

sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));


% Los parametros de la red neuronal
num_entradas  = 400;  
num_ocultas = 25;  
num_etiquetas = 10;     

% Desenrollamos los vectores theta.
params_rn = [Theta1(:) ; Theta2(:)];

%Usamos lambda = 0 para no regularizar
lambda = 0;

J = costeRN(params_rn, num_entradas, num_ocultas, ...
                   num_etiquetas, X, y, lambda);
fprintf('El costo sin regularizacion es %f ', J);
fprintf('\n\n');


%Calculo de coste con reguralizacion:

% usamos 1 como valor de regularizacion 
lambda = 1;

J = costeRN(params_rn, num_entradas, num_ocultas, ...
                   num_etiquetas, X, y, lambda);

fprintf('El costo CON regularizacion es %f ', J);
fprintf('\n\n');

% El gradiente sigmoide
g = sigmoideGradiente([1 -0.5 0 0.5 1]);
fprintf('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n  ');
fprintf('%f ', g);
fprintf('\n\n');




%Iniciamos los valores iniciales de Theta con valores aleatorios:

theta1Inicial = pesosAleatorios(num_entradas, num_ocultas);
theta2Inicial = pesosAleatorios(num_ocultas, num_etiquetas);

% desenrollamos los parametros
params_rn_iniciales = [theta1Inicial(:) ; theta2Inicial(:)];



%Utilizamos la funcion checkNNGradientes para comprobar
checkNNGradients;


%  probamos los gradientes con lambda = 3
lambda = 3;
checkNNGradients(lambda);

% muestra los valores de debug para la funcion
J = costeRN(params_rn, num_entradas, num_ocultas, ...
                   num_etiquetas, X, y, lambda);

fprintf(['\n\nCost at (fixed) debugging parameters (w/ lambda = 10): %f ' ...
         '\n(this value should be about 0.576051)\n\n'], J);


%Ahora pasamos a entrenar la red neuronal

options = optimset('MaxIter', 50);

lambda = 1;


funcionCoste = @(p)costeRN(params_rn, num_entradas, num_ocultas, ...
                   num_etiquetas, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[params_rn, coste] = fmincg(funcionCoste, params_rn_iniciales, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(params_rn(1:num_ocultas * (num_entradas + 1)), ...
                 num_ocultas, (num_entradas + 1));

Theta2 = reshape(params_rn((1 + (num_ocultas * (num_entradas + 1))):end), ...
                 num_etiquetas, (num_ocultas + 1));



displayData(Theta1(:, 2:end));

pred = predecir(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);



