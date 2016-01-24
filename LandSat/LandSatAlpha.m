%Proyecto Aprendizaje Automatico y Big Data
%Emanuel Ramirez Catapano
%Multi-class Logistic Regression
%Realizado con datos cargados de Landsat statlog en sat_trn y sat_tst.txt


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

%displayData(sel_red,autumn);
% figure;
%displayData(sel_green,summer);
% figure;
% displayData(sel_infra1,gray);
% figure;
% displayData(sel_infra2,gray);


%=============== Multiclass Logistic Regression ===========================

% [all_theta] = oneVsAll(training,Y,6,1);
% [p] = predictOneVsAll(all_theta,test);
% acertados = p==Y_test;
% fprintf('\nEl porcentaje de aciertos: %f\n', (sum(acertados)/...
%     size(acertados,1))*100);
% fprintf('\nPresiona enter para continuar.');
% pause;

%=============== Support Vector Machines ==================================
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
num_entradas  = 36;  
num_ocultas = 21;   
num_etiquetas = 6;

fprintf('\nIniciamos pesos aleatorios para los parametros\n')

theta1_inicial = pesosAleatorios(num_entradas, num_ocultas);
theta2_inicial = pesosAleatorios(num_ocultas, num_etiquetas);


% hablar sobre regularizacion de parametros /255 
% desenrollamos parametros
params_rn_inicial = [theta1_inicial(:) ; theta2_inicial(:)];
testing_iter = [50,70,100,150,200,300];
testing_lambda = [0,0.1,0.3,1,3,10,30];
fprintf('\nEntrenamiento de red neuronal \n');
results = zeros(length(testing_iter),length(testing_lambda));
for i=1:length(testing_iter)
    options = optimset('MaxIter',testing_iter(i));
    for j=1:length(testing_lambda)
        costFunction = @(p) costeRN(p, ...
                                           num_entradas, ...
                                           num_ocultas, ...
                                           num_etiquetas, training/255, Y, testing_lambda(j));
        [params_rn, cost] = fmincg(costFunction, params_rn_inicial, options);

        % Obtenemos theta de params_rn
        Theta1 = reshape(params_rn(1:num_ocultas * (num_entradas + 1)), ...
                         num_ocultas, (num_entradas + 1));

        Theta2 = reshape(params_rn((1 + (num_ocultas * (num_entradas + 1))):end), ...
                         num_etiquetas, (num_ocultas + 1));

        % fprintf('Presiona Enter');
        % pause;
        % 
        % fprintf('\nMostrando pesos de la Red... \n')

        %displayData(Theta1(:, 2:end));


        % fprintf('\nPresione Enter para continuar.\n');
        % pause;

        pred = predecir(Theta1, Theta2, test/255);
        results(i,j) = mean(double(pred == Y_test)) * 100;
    end
end
fprintf('\nLa tasa de aciertos es:  %f\n', mean(double(pred == Y_test)) * 100);



%=============== Clustering ===============================================

%Para esta parte intento tomar los datos como un problema de aprendizaje
%no supervisado, para ver si los ejemplos forman algun tipo de cluster y
%luego comparare con las prediccion de validacion.



