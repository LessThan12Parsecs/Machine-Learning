%Proyecto Aprendizaje Automatico y Big Data
%Emanuel Ramirez Catapano
%Multi-class Logistic Regression
%Realizado con datos cargados de Landsat statlog en sat_trn y sat_tst.txt
%no se realizara cross validation por comentario del proveedor de los datos
%en la descripcion. 

%=============== Cargamos los datos en X e Y de entrenmiento y test========
training = importdata('sat_trn.txt');
test = importdata('sat_tst.txt');
Y = training(:,37);
Y_test = test(:,37);
test(:,37) = [];
training(:,37) = [];

%cambio los valores que sean 7 por 6, ya que la clase 6 no existe por
%limitaciones de los ejemplos de datos. 
Y(Y==7) = 6;
Y_test(Y_test==7)=6;

%divido en en una matriz para cada espectro de imagen.
% X_green = training(:,1:4:end);
% X_red = training(:,2:4:end);
% X_infra1= training(:,3:4:end);
% X_infra2= training(:,4:4:end);
%tomamos solo los pixeles centrales para pruebas de efectividad.
X_cen_trn = training(:,17:20);
X_cen_tst = test(:,17:20);
X_cen_trn_reg = X_cen_trn/255;
X_cen_tst_reg = X_cen_tst/255;



% m =size(training,1);
% rand_indices = randperm(m);
% sel_green = X_green(rand_indices(1:100),:);
% sel_red = X_red(rand_indices(1:100),:);
% sel_infra1 = X_infra1(rand_indices(1:100),:);
% sel_infra2 = X_infra2(rand_indices(1:100),:);

% displayData(sel_red,autumn); %esta funcion fue modicada para los datos
%  figure;
% displayData(sel_green,summer);
%  figure;
%  displayData(sel_infra1,gray);
%  figure;
%  displayData(sel_infra2,gray);


%=============== Regresion Logistica Multiclase ===========================
%Falta ajustar.

[all_theta] = oneVsAll(X_cen_trn_reg,Y,6,1);
[p] = predictOneVsAll(all_theta,X_cen_tst_reg);
acertados = p==Y_test;
fprintf('\nEl porcentaje de aciertos: %f\n', (sum(acertados)/...
    size(acertados,1))*100);
fprintf('\nPresiona enter para continuar.');
pause;
%Aplicando regularizacion en el rango de los atributos.
 training_reg = training/255;
 test_reg = test/255;
% [all_theta] = oneVsAll(training_reg,Y,6,1);
% [p] = predictOneVsAll(all_theta,test_reg);
% acertados = p==Y_test;
% fprintf('\nEl porcentaje de aciertos con regularizacion: %f\n',...
%     (sum(acertados)/size(acertados,1))*100);
% fprintf('\nPresiona enter para continuar.');
% pause;

%=============== Support Vector Machines ==================================
% hay que probar con oneVsAll para el SVM
% valores = [0.3,1];
% percent = zeros(2,1);
% 
% training = training/255;
% test = test/255;
% actual = Y==1; 
% for iC = 1:length(valores)
%         model = svmTrain(training,actual,valores(iC),@(x1,x2) gaussianKernel(x1,x2,1));
%         prediction = svmPredict(model,test);
%         success = (prediction == Y_test);
%         percent(iC,1) = sum(success)/length(success);
% end


%plotMultiClass(training(:,17:20),Y,6);




%=============== Neural Networks ==========================================


%un 81% de aciertos, falta mejorar
num_entradas  = 4;  
num_ocultas = 12;
num_etiquetas = 6;

num_ocultasMax

fprintf('\nIniciamos pesos aleatorios para los parametros\n')

theta1_inicial = pesosAleatorios(num_entradas, num_ocultas);
theta2_inicial = pesosAleatorios(num_ocultas, num_etiquetas);

% usa gradient Checking antes de entrenar solamente.
checkNNGradients;

% desenrollamos parametros
params_rn_inicial = [theta1_inicial(:) ; theta2_inicial(:)];
lambda = 0;

fprintf('\nEntrenamiento de red neuronal \n');

for i = 1:num_oculta
options = optimset('MaxIter',70);
costFunction = @(p) costeRN(p,num_entradas,num_ocultas,num_etiquetas,...
    X_cen_trn_reg, Y, lambda);
[params_rn, cost] = fmincg(costFunction, params_rn_inicial, options);


% Entrenamos tambien con los de validacion como prueba
% costFunction = @(p) costeRN(p,num_entradas,num_ocultas,num_etiquetas,...
%     test_reg, Y_test, lambda);
% [params_rn_test, cost_test] = fmincg(costFunction, params_rn_inicial,...
%     options);



% Obtenemos theta de params_rn
Theta1 = reshape(params_rn(1:num_ocultas * (num_entradas + 1)), ...
                                 num_ocultas, (num_entradas + 1));

Theta2 = reshape(params_rn((1 + (num_ocultas * (num_entradas + 1))):end),...
                                 num_etiquetas, (num_ocultas + 1));


pred = predecir(Theta1, Theta2, X_cen_tst_reg);
resultado = mean(double(pred == Y_test)) * 100;
fprintf('\nLa tasa de aciertos es:  %f\n', resultado);



%=============== Clustering ===============================================

%Para esta parte intento tomar los datos como un problema de aprendizaje
%no supervisado, para ver si los ejemplos forman algun tipo de clusters y
%luego comparare con las prediccion de validacion

% randidx = randperm(size(training,1));
% centroides = training(randidx(1:6),:);
% runkMeans(training,centroides,10,true);

