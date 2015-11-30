%Practica 4 - Aprendizaje Automatico y Big Data
%Emanuel Ram?rez Catapano
%Entrenamientos de Redes Neuronales
%Realizado cargando los datos de ex4data1.mat y ex4weights.mat



% Damos Valores para el tama?o de nuestra red neuronal
num_entradas  = 400;  
num_ocultas = 25;   
num_etiquetas = 10;             
                          

% =========== Primera Parte: visualizacion de datos =============


% Cargamos los datos de entrenamiento 
fprintf('La visualizacion de los datos: ')
load('ex4data1.mat');
m = size(X, 1);


sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Presiona Enter');
pause;


% Cargamos los pesos en Theta1 y Theta2
load('ex4weights.mat');

% Los desenrollamos 
params_rn = [Theta1(:) ; Theta2(:)];

% ================ Part 3: Compute Cost (Feedforward) ================

fprintf('\nRealizamos la propagacion para el calculo del coste: \n')

% Weight regularization parameter (we set this to 0 here).
lambda = 0;

J = costeRN(params_rn, num_entradas, num_ocultas, ...
                   num_etiquetas, X, y, lambda);

fprintf(['El coste con los parametros cargados de ex4weights: %f '], J);

fprintf('Presiona Enter');
pause;

% =============== Part 4: Implement Regularization ===============
%  Once your cost function implementation is correct, you should now
%  continue to implement the regularization with the cost.
%

fprintf('\nCalculamos el coste con reguralizacion\n')

% Con el parametro de regularizacion puesto a 1.
lambda = 1;

J = costeRN(params_rn, num_entradas, num_ocultas, ...
                   num_etiquetas, X, y, lambda);

fprintf(['El costo con los parametros cargados de ex4weights y con reguralizacion es: %f '], J);

fprintf('Presiona Enter');
pause;


% ================ Part 5: Sigmoid Gradient  ================

fprintf('\nEvaluamos el Sigmoide Gradiente')

g = sigmoideGradiente([1 -0.5 0 0.5 1]);
fprintf('El sigmoide gradiente evaluado en [1 -0.5 0 0.5 1]:\n  ');
fprintf('%f ', g);
fprintf('\n\n');

fprintf('Presiona Enter');
pause;


% ================ Part 6: Initializing Pameters ================


fprintf('\nIniciamos pesos aleatorios para los parametros\n')

theta1_inicial = pesosAleatorios(num_entradas, num_ocultas);
theta2_inicial = pesosAleatorios(num_ocultas, num_etiquetas);

% desenrollamos parametros
params_rn_inicial = [theta1_inicial(:) ; theta2_inicial(:)];


% =============== Part 7: Implement Backpropagation ===============

fprintf('\nImplementando Retro-Propagacion \n');

%  probamos con la funcion checkNNGradients
checkNNGradients;
fprintf('Presiona Enter');
pause;



% =============== Part 8: Implement Regularization ===============


fprintf('\nRetro-propagacion con reguralizacion ... \n')

%  volvemos a comprobar con checkNNGradients y lambda = 3
lambda = 3;
checkNNGradients(lambda);


% 
% % mostramos los valores de debug para el coste
% debug_J  = costeRN(params_rn, num_entradas, ...
%                           num_ocultas, num_etiquetas, X, y, lambda);
% 
% fprintf(['\n\nCost at (fixed) debugging parameters (w/ lambda = 10): %f '], debug_J);
% 
% fprintf('Presiona Enter');
% pause;


% =================== Part 8: Training NN ===================


fprintf('\nEntrenamiento de red neuronal \n')

options = optimset('MaxIter', 50);
lambda = 1;
costFunction = @(p) costeRN(p, ...
                                   num_entradas, ...
                                   num_ocultas, ...
                                   num_etiquetas, X, y, lambda);
[params_rn, cost] = fmincg(costFunction, params_rn_inicial, options);

% Obtenemos theta de params_rn
Theta1 = reshape(params_rn(1:num_ocultas * (num_entradas + 1)), ...
                 num_ocultas, (num_entradas + 1));

Theta2 = reshape(params_rn((1 + (num_ocultas * (num_entradas + 1))):end), ...
                 num_etiquetas, (num_ocultas + 1));

fprintf('Presiona Enter');
pause;


% ================= Part 9: Visualize Weights =================
%  You can now "visualize" what the neural network is learning by 
%  displaying the hidden units to see what features they are capturing in 
%  the data.

fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end));


fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% ================= Part 10: Implement Predict =================


pred = predecir(Theta1, Theta2, X);

fprintf('\nLa tasa de aciertos es:  %f\n', mean(double(pred == y)) * 100);

