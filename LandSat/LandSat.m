%Proyecto Aprendizaje Automatico y Big Data
%Emanuel Ramirez Catapano
%Realizado con datos cargados de Landsat statlog en sat_trn y sat_tst.txt
%no se realizara cross validation por comentario del proveedor de los datos
%en la descripcion. 

%=============== Cargamos los datos en X e Y de entrenmiento y test========
training = importdata('sat_trn.txt');
test = importdata('sat_tst.txt');
Y = training(:,37);
Y_test = test(:,37);
training(:,37) = [];
test(:,37) = [];
%cambio los valores que sean 7 por 6, ya que la clase 6 no existe por
%limitaciones de los ejemplos de datos. 
Y(Y==7) = 6;
Y_test(Y_test==7)=6;

% para el uso en la app learner de matlab
XApp = [training,Y; test,Y_test];


training = featureNormalize(training/255);
test = featureNormalize(test/255);
%tomamos solo los pixeles centrales para pruebas de efectividad.
X_cenTrn = training(:,17:20);
X_cenTst = test(:,17:20);

% %divido en en una matriz para cada espectro de imagen.
X_green = training(:,1:4:end);
X_red = training(:,2:4:end);
X_infra1= training(:,3:4:end);
X_infra2= training(:,4:4:end);

% tomo indices aleatorios para hacer una seleccion que graficar como
% prueba.
m =size(training,1);
rand_indices = randperm(m);
sel_green = X_green(rand_indices(1:100),:);
sel_red = X_red(rand_indices(1:100),:);
sel_infra1 = X_infra1(rand_indices(1:100),:);
sel_infra2 = X_infra2(rand_indices(1:100),:);


% Utilizo la funcion displayData para mostrar los cuadros de 3x3 pixeles
% representados por separados en las 4 bandas espectrales y con tonos de 0
% a 255 en intensidad. 

displayData(sel_red,autumn); %esta funcion fue modicada para los datos
figure;
displayData(sel_green,summer);
figure;
displayData(sel_infra1,bone);
figure;
displayData(sel_infra2,pink);


%=============== Regresion Logistica Multiclase ===========================
%Falta ajustar.
lambdas = [0,0.1,0.3,10,40000,100000];
resultadosTst = zeros(6,1);
resultadosTrn= zeros(6,1);

for i = 1:length(lambdas)
    [all_theta] = oneVsAll(training,Y,6,lambdas(i));
    p = predictOneVsAll(all_theta,training);
    p1 = predictOneVsAll(all_theta,test);
    acertadosTrn = p==Y;
    acertadosTst = p1==Y_test;
    resultadosTrn(i,1) = (sum(acertadosTrn)/size(acertadosTrn,1));
    resultadosTst(i,1) =(sum(acertadosTst)/size(acertadosTst,1));
end 

plot(lambdas,resultadosTst);
xlabel('lambdas');
ylabel('aciertos');
hold on
plot(lambdas,resultadosTrn);
grid('on');
legend('Test','Training');
fprintf('\nEl mejor resultado en Validacion es de: %f', max(resultadosTst));
fprintf('\nEl mejor resultado en Entrenamiento es de: %f', max(resultadosTrn));
pause;

%=============== Support Vector Machines ==================================
%Realizado en el app de Matlab.
[mdl,percent] = gaussianSVMTrain(XApp);
%=============== Neural Networks ==========================================

%un 81% de aciertos
num_entradas  = 36;
lambdas = [0,0.1,0.3,1,3,10];
num_ocultas = [4,10,15,20,30,50];
num_etiquetas = 6;
resultados = zeros(length(lambdas),length(num_ocultas));
resultadosTest = zeros(length(lambdas),length(num_ocultas));

%usa gradient Checking antes de entrenar solamente.
checkNNGradients;
fprintf('\nEntrenamiento de red neuronal \n');
tic;
for i=1:length(num_ocultas)
    for j=1:length(lambdas)
        theta1_inicial = pesosAleatorios(num_entradas, num_ocultas(i));
        theta2_inicial = pesosAleatorios(num_ocultas(i), num_etiquetas);
        params_rn_inicial = [theta1_inicial(:) ; theta2_inicial(:)];
        options = optimset('MaxIter',70);
        costFunction = @(p) costeRN(p,num_entradas,num_ocultas(i),num_etiquetas,...
            training, Y, lambdas(i));
        [params_rn, cost] = fmincg(costFunction, params_rn_inicial, options);
        Theta1 = reshape(params_rn(1:num_ocultas(i) * (num_entradas + 1)), ...
                                 num_ocultas(i), (num_entradas + 1));
        Theta2 = reshape(params_rn((1 + (num_ocultas(i) * (num_entradas + 1))):end),...
                                 num_etiquetas, (num_ocultas(i) + 1));
        pred = predecir(Theta1, Theta2, training);
        pred2 = predecir(Theta1,Theta2,test);
        resultados(i,j) = mean(double(pred == Y)) * 100;
        resultadosTest(i,j) = mean(double(pred2 == Y_test)) * 100;
    end
end
time = toc;
fprintf('en un tiempof de %f',time);
plot(lambdas,resultados);
legend('4 Hidden','10 Hidden','15 Hidden','20 Hidden','30 Hidden','50 Hidden');
TableTrn = array2table(resultados,'RowNames',...
    {'lambda 0','lambda 0.1','lambda 0.3','lambda 1','lambda 3','lambda 10'});
%=============== Classification Trees =====================================
%usamos la funcion del toolbox de matlab para el entrenamiento de un arbol
%de clasificiacion 
% 
tree = fitctree(training,Y);
prediTree = predict(tree,test);
resultado = mean(double(prediTree == Y_test)) * 100;
view(tree,'Mode','graph');
fprintf('\n La tasa de aciertos del arbol de decision es: %f',resultado);
%probamos una tecnica de Tree ensemble o "random forest"
BaggedEnsemble = TreeBagger(50,training,Y,'OOBPred','On');
oobErrorBaggedEnsemble = oobError(BaggedEnsemble);
plot(oobErrorBaggedEnsemble)
xlabel ('Numero de Arboles');
ylabel ('Error de clasificacion "out-of-bag"');
prediction = BaggedEnsemble.predict(test);
resultado2 = mean(double(cell2mat(prediction) == Y_test)) * 100;


%=============== K-NearNeighbors===========================================
% Realizado con el Classification Learner de Matlab
[mdl,percentK] = kNearNeighbors(XApp);





