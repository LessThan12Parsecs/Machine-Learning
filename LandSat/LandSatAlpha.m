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
%escalamos los datos para que esten entre 0 y 1
% training = training/255;
% test = test/255;

%cambiamos los valores que sean 7 por 6, ya que la clase 6 no existe por
%limitaciones de los ejemplos de datos. 
Y(Y==7) = 6;
Y_test(Y_test==7)=6;

%dividimos en en una matriz para cada espectro de imagen.
X_green = training(:,1:4:end);
X_red = training(:,2:4:end);
X_infra1= training(:,3:4:end);
X_infra2= training(:,4:4:end);
%tomamos solo los pixeles centrales para pruebas de efectividad.
X_cen_trn = training(:,17:20);
X_cen_tst = test(:,17:20);



m =size(training,1);
rand_indices = randperm(m);
sel_green = X_green(rand_indices(1:100),:);
sel_red = X_red(rand_indices(1:100),:);
sel_infra1 = X_infra1(rand_indices(1:100),:);
sel_infra2 = X_infra2(rand_indices(1:100),:);

%displayData(sel_red,autumn); esta funcion fue modicada para los datos
% figure;
%displayData(sel_green,summer);
% figure;
% displayData(sel_infra1,gray);
% figure;
% displayData(sel_infra2,gray);


%=============== Multiclass Logistic Regression ===========================
%implementacion bastante basica pero funciona mas o menos bien.

% [all_theta] = oneVsAll(training,Y,6,1);
% [p] = predictOneVsAll(all_theta,test);
% acertados = p==Y_test;
% fprintf('\nEl porcentaje de aciertos: %f\n', (sum(acertados)/...
%     size(acertados,1))*100);
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


% %un 81% de aciertos, falta mejorar
% num_entradas  = 36;  
% num_ocultas = 42;   
% num_etiquetas = 6;
% 
% %aplicamos regularizacion
% training = training/255;
% test = test/255;
% fprintf('\nIniciamos pesos aleatorios para los parametros\n')
% 
% theta1_inicial = pesosAleatorios(num_entradas, num_ocultas);
% theta2_inicial = pesosAleatorios(num_ocultas, num_etiquetas);
% 
% % usa gradient Checking antes de entrenar solamente.
% 
% % hablar sobre regularizacion de parametros /255 
% % desenrollamos parametros
% params_rn_inicial = [theta1_inicial(:) ; theta2_inicial(:)];
% lambda = 1;
% fprintf('\nEntrenamiento de red neuronal \n');
% checkNNGradients;
% options = optimset('MaxIter',70);
% costFunction = @(p) costeRN(p,num_entradas,num_ocultas,num_etiquetas,...
%     training, Y, lambda);
% [params_rn, cost] = fmincg(costFunction, params_rn_inicial, options);
% 
% costFunction = @(p) costeRN(p,num_entradas,num_ocultas,num_etiquetas,...
%     test, Y_test, lambda);
% [params_rn_test, cost_test] = fmincg(costFunction, params_rn_inicial, options);
% 
% % Obtenemos theta de params_rn
% Theta1 = reshape(params_rn(1:num_ocultas * (num_entradas + 1)), ...
%                                  num_ocultas, (num_entradas + 1));
% 
% Theta2 = reshape(params_rn((1 + (num_ocultas * (num_entradas + 1))):end), ...
%                                  num_etiquetas, (num_ocultas + 1));
% 
% 
% Theta11 = Theta1(:);                             
% rand_indices = randperm(length(Theta1(:)));
% sel_Theta1 = Theta11(rand_indices(1:70),:);
% plot(cost,sel_Theta1);
% % fprintf('Presiona Enter');
% % pause;
% % fprintf('\nMostrando pesos de la Red... \n')
% %displayData(Theta1(:, 2:end));
% % fprintf('\nPresione Enter para continuar.\n');
% % pause;
% pred = predecir(Theta1, Theta2, test);
% resultado = mean(double(pred == Y_test)) * 100;
% fprintf('\nLa tasa de aciertos es:  %f\n', resultado);



%=============== Clustering ===============================================

%Para esta parte intento tomar los datos como un problema de aprendizaje
%no supervisado, para ver si los ejemplos forman algun tipo de clusters y
%luego comparare con las prediccion de validacion

randidx = randperm(size(training,1));
centroides = training(randidx(1:6),:);
runkMeans(training,centroides,10,true);

