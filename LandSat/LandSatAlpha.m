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
training(:,37) = [];
Y_test = test(:,37);
test(:,37) = [];

%cambio los valores que sean 7 por 6, ya que la clase 6 no existe por
%limitaciones de los ejemplos de datos. 
Y(Y==7) = 6;
Y_test(Y_test==7)=6;


%tomamos solo los pixeles centrales para pruebas de efectividad.
X_cen_trn = training(:,17:20);
X_cen_tst = test(:,17:20);
X_cen_trn_reg = X_cen_trn/255;
X_cen_tst_reg = X_cen_tst/255;

X_cenTrn = featureNormalize(X_cen_trn_reg);
X_cenTst = featureNormalize(X_cen_tst_reg);


%divido en en una matriz para cada espectro de imagen.
%X_green = training(:,1:4:end);
%X_red = training(:,2:4:end);
% X_infra1= training(:,3:4:end);
% X_infra2= training(:,4:4:end);

% tomo indices aleatorios para hacer una seleccion que graficar como
% prueba.
% m =size(training,1);
% rand_indices = randperm(m);
% sel_green = X_green(rand_indices(1:100),:);
% sel_red = X_red(rand_indices(1:100),:);
% sel_infra1 = X_infra1(rand_indices(1:100),:);
% sel_infra2 = X_infra2(rand_indices(1:100),:);


% Utilizo la funcion displayData para mostrar los cuadros de 3x3 pixeles
% representados por separados en las 4 bandas espectrales y con tonos de 0
% a 255 en intensidad. 

% displayData(sel_red,autumn); %esta funcion fue modicada para los datos
% figure;
% displayData(sel_green,summer);
% figure;
% displayData(sel_infra1,gray);
% figure;
% displayData(sel_infra2,gray);


%=============== Regresion Logistica Multiclase ===========================
%Falta ajustar.
 lambdas = [0,0.01,0.03,0.1];
% resultados = zeros(6,1);
% resultadosTrn= zeros(6,1);
% 
% for i = 1:length(lambdas)
%     [all_theta] = oneVsAll(X_cenTrn,Y,6,lambdas(i));
%     [p] = predictOneVsAll(all_theta,X_cenTst);
%     [p1] = predictOneVsAll(all_theta,X_cenTrn);
%     acertados = p==Y_test;
%     acertadosTrn = p1==Y;
%     resultados(i,1) = 1 -(sum(acertados)/size(acertados,1));
%     resultadosTrn(i,1) = 1 - (sum(acertadosTrn)/size(acertadosTrn,1));
%     %fprintf('\nEl porcentaje de aciertos: %f\n', (sum(acertados)/...
%        % size(acertados,1))*100);
%   %  fprintf('\nPresiona enter para continuar.');
%    % pause;
% end
% 
% 
% plot(lambdas,resultados);
% xlabel('lambdas');
% ylabel('error');
% hold on
% plot(lambdas,resultadosTrn);
% grid('on');
% legend('Validation','Training');
% fprintf('\nEl menor error en Validacion es de: %f', min(resultados));
% fprintf('\nEl menor error en entrenamiento es de: %f', min(resultadosTrn));
% pause;
%Aplicando regularizacion en el rango de los atributos.
%training_reg = training/255;
%test_reg = test/255;
% [all_theta] = oneVsAll(training_reg,Y,6,1);
% [p] = predictOneVsAll(all_theta,test_reg);
% acertados = p==Y_test;
% fprintf('\nEl porcentaje de aciertos con regularizacion: %f\n',...
%     (sum(acertados)/size(acertados,1))*100);
% fprintf('\nPresiona enter para continuar.');
% pause;

%=============== Support Vector Machines ==================================

%=============== Neural Networks ==========================================

% 
%un 81% de aciertos, falta mejorar
num_entradas  = 4;  
num_ocultas = [4,10,15,20];
num_etiquetas = 6;
resultados = zeros(length(lambdas),length(num_ocultas));

% usa gradient Checking antes de entrenar solamente.
 checkNNGradients;

fprintf('\nEntrenamiento de red neuronal \n');
for i=1:length(num_ocultas)
    for j=1:length(lambdas)
        theta1_inicial = pesosAleatorios(num_entradas, num_ocultas(i));
        theta2_inicial = pesosAleatorios(num_ocultas(i), num_etiquetas);
        params_rn_inicial = [theta1_inicial(:) ; theta2_inicial(:)];
        options = optimset('MaxIter',70);
        costFunction = @(p) costeRN(p,num_entradas,num_ocultas(i),num_etiquetas,...
            X_cenTrn, Y, lambdas(i));
        [params_rn, cost] = fmincg(costFunction, params_rn_inicial, options);
        Theta1 = reshape(params_rn(1:num_ocultas(i) * (num_entradas + 1)), ...
                                 num_ocultas(i), (num_entradas + 1));
        Theta2 = reshape(params_rn((1 + (num_ocultas(i) * (num_entradas + 1))):end),...
                                 num_etiquetas, (num_ocultas(i) + 1));
        pred = predecir(Theta1, Theta2, X_cenTrn);
        resultados(i,j) = mean(double(pred == Y)) * 100;
    end
end
[max,index] = max(resultados(:));
fprintf('\nLa tasa mayor de aciertos es:  %f  para lambda: %f y con %i capas ocultas\n',max,lambdas(round(index/4)),num_ocultas(mod(index,4)));


%=============== Classification Trees =====================================
%usamos la funcion del toolbox de matlab para el entrenamiento de un arbol
%de clasificiacion 
% 
% tree = fitctree(X_cenTrn,Y);
% prediTree = predict(tree,X_cenTst);
% resultado = mean(double(prediTree == Y_test)) * 100;
% view(tree,'Mode','graph');
% fprintf('\n La tasa de aciertos del arbol de decision es: %f',resultado);

%=============== Clustering ===============================================

%Para esta parte intento tomar los datos como un problema de aprendizaje
%no supervisado, para ver si los ejemplos forman algun tipo de clusters y
%luego comparare con las prediccion de validacion

% randidx = randperm(size(training,1));
% centroides = training(randidx(1:6),:);
% runkMeans(training,centroides,10,true);

