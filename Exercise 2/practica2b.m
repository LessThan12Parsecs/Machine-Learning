%Emanuel Ramirez Catapano
%Practica 2 Aprendizaje Automatico
%Regresion Logistica con regularizacion
%Datos tomados de ex2data2.txt

%Abrimos el archivo
fileID = fopen('ex2data2.txt'); 
C = textscan(fileID,'%f,%f,%f');
fclose(fileID);

X = [C{1},C{2}];
Y = [C{3}];

%empezamos por graficar los puntos negativos
X = [ones(size(X(:,1))),X];
negativos = find(Y==0);
plot(X(negativos,2),X(negativos,3),'ko','MarkerFaceColor','r','MarkerSize',7);
hold on
positivos = find(Y==1);
plot(X(positivos,2),X(positivos,3),'d','MarkerFaceColor','g','MarkerSize',6);
legend('Insuficiente Calidad','Suficiente Calidad');
xlabel('Microchip test 1');ylabel('Microchip test 2');

%extendemos los ejemplos de entrenamiento con mas atributos.
X = mapFeature(X(:,2),X(:,3));

n = length(X(1,:));

%inicializamos theta a zero
theta_inicial = zeros(n,1);
lambda = [0,1,2,3];
options = (optimset('GradObj','on','MaxIter',400));

for i = 1:length(lambda);
    [theta,cost] = fminunc(@(t)(costeRegularizado(lambda(i),t,X,Y)),theta_inicial,options);
    [porcentajeRegul] = evaluacionRegLog(theta,X,Y)
    plotDecisionBoundary(theta,X,Y);
end
    
