%Practica 6a Aprendizaje Automatico y Big Data
%Emanuel Ramirez Catapano
%Prueba de Support Vector Machines
%Realizado con los datos cargados de ex6data1-2-3.mat

%Cargamos los datos para frontera lineal.
load('ex6data1');
%plotData(X,y);

%entrenamos la svm con el kernel lineal con C = 1
model = svmTrain(X,y,1, @linearKernel,1e-3,20);

%visualizamos la frontera
visualizeBoundaryLinear(X,y,model);

%ahora con C = 100
model = svmTrain(X,y,100, @linearKernel,1e-3,20);

visualizeBoundaryLinear(X,y,model);

load('ex6data2.mat');
plotData(X,y);

%para el calculo de la frontera no lineal, entrenamos con sigma = 0,1 y C=1 
sigma = 0.1; C = 1;
model = svmTrain(X,y,C,@(x1,x2) gaussianKernel(x1,x2,sigma));
visualizeBoundary(X,y,model);


%Eleccion de parametros C y sigma

load('ex6data3.mat');
valores = [0.01,0.03,0.1,0.3,1,3,10,30];
percent = zeros(8,8);

for iC = 1:length(valores)
    for iSigma = 1:length(valores)
        model = svmTrain(X,y,valores(iC),@(x1,x2) gaussianKernel(x1,x2,valores(iSigma)));
        prediction = svmPredict(model,Xval);
        success = (prediction == yval);
        percent(iC,iSigma) = sum(success)/length(success);
    end
end

T = array2table(percent,'RowNames',{'0.01','0.03','0.1','0.3','1','3','10','30'}...
    ,'VariableNames',{'c0_01','c0_03','c0_1','c0_3','c1','c3','c10','c30'});






