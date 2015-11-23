%Emanuel Ramirez Catapano
%Practica 2 Aprendizaje Automatico
%Regresion Logistica
%Datos tomados de ex2data1.txt


%Abrimos el archivo
fileID = fopen('ex2data1.txt');
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
legend('No Admitidos','Admitidos');
xlabel('Nota en 1ra prueba');ylabel('Nota en 2da prueba');

n = length(X(1,:));

%inicializamos theta a zero
theta_inicial = zeros(n,1);

%Para el calculo del valor minimo de theta usamos la funcion fminunc
%tambien disponible en matlab.

options = (optimset('GradObj','on','MaxIter',400));

%obtenemos el valor optimo de theta

[theta,cost] = fminunc(@(t)(coste(t,X,Y)),theta_inicial,options);

%graficamos la utilizando la funcion plotDecisionBoundary con los valores
%nuevos de theta

plotDecisionBoundary(theta,X,Y);
[porcentaje] = evaluacionRegLog(theta,X,Y)