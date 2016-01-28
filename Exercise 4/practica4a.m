%Practica 4 - Aprendizaje Automatico y Big Data
%Emanuel Ram?rez Catapano
%Entrenamientos de Redes Neuronales
%Realizado cargando los datos de ex4data1.mat y ex4weights.mat



% Damos Valores para el tama?o de nuestra red neuronal
        
                          

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

% ================ Coste (Feedforward) ====================================

fprintf('\nRealizamos la propagacion para el calculo del coste: \n')

% eleccion de Lambda para regularizacion
lambda = 0;

J = costeRN(params_rn, num_entradas, num_ocultas, ...
                   num_etiquetas, X, y, lambda);

fprintf(['El coste con los parametros cargados de ex4weights: %f '], J);

fprintf('Presiona Enter');
pause;

% =============== Implementacion de la regularizacion =====================

fprintf('\nCalculamos el coste con reguralizacion\n')

% Con el parametro de regularizacion puesto a 1.
lambda = 1;

J = costeRN(params_rn, num_entradas, num_ocultas, ...
                   num_etiquetas, X, y, lambda);

fprintf(['El costo con los parametros cargados de ex4weights y con reguralizacion es: %f '], J);
fprintf('Presiona Enter');
pause;


% ================Gradiente sigmoide  =====================================

fprintf('\nEvaluamos el Sigmoide Gradiente')

g = sigmoideGradiente([1 -0.5 0 0.5 1]);
fprintf('El sigmoide gradiente evaluado en [1 -0.5 0 0.5 1]:\n  ');
fprintf('%f ', g);
fprintf('\n\n');

fprintf('Presiona Enter');
pause;


% ================ Inicializamos los parametros ===========================


fprintf('\nIniciamos pesos aleatorios para los parametros\n')

theta1_inicial = pesosAleatorios(num_entradas, num_ocultas);
theta2_inicial = pesosAleatorios(num_ocultas, num_etiquetas);

% desenrollamos parametros
params_rn_inicial = [theta1_inicial(:) ; theta2_inicial(:)];


% ================ Backpropagation ========================================

fprintf('\nImplementando Retro-Propagacion \n');

%  probamos con la funcion checkNNGradients
checkNNGradients;
fprintf('Presiona Enter');
pause;



% =============== Regularizacion ==========================================


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


% =================== Entramiento de la Red Neuronal=======================


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


% ================= Visualizamos los pesos de la capa oculta ==============


displayData(Theta1(:, 2:end));


fprintf('\nPresione enter\n');
pause;

% ================= Implementamos las Predicciones  =======================


pred = predecir(Theta1, Theta2, X);

fprintf('\nLa tasa de aciertos es:  %f\n', mean(double(pred == y)) * 100);

