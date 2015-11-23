%Practica 1 Aprendizaje Automatico
%Emanuel Ramirez Catapano
%Regresion Lineal, una variable.
%Cargando datos de archivo ex1data1.txt



fileID = fopen('ex1data1.txt'); % Abrimos el archivo y lo asignamos a un ID.
C  = textscan(fileID,'%f , %f'); % cargamos los datos en un cell C.
fclose(fileID); % cerramos el archivo.

m = length(C{2}); % calculamos el numero de ejemplo de entrenamiento.
figure; %Abrimos una nueva figura.
plot(C{1},C{2},'o'); % graficamos los datos de entrenamiento. 
xlabel('Habitantes expresados en 10mil');ylabel('Beneficio expresado en 10mil $'); % nombramos los ejes del planos
Mx = [ones(m,1),C{1}]; %Pasamos la primera columna a forma matriz y a?adimos unos a la primera componente.
My = [C{2}];%cambiamos la columna de salidas a forma vectorial.

theta = zeros(size(Mx(1,:)))'; % inicializamos los parametros theta.
NUM_ITER = 1500; % numero de iteraciones 
alpha = 0.01; % damos un valor al coeficiente de aprendizaje.

%iniciamos las iteraciones
for num = 1:NUM_ITER
    %El calculo del gradiente
    gradi = (1/m).* Mx' * ((Mx * theta) - My);
    theta = theta - alpha.*gradi; % actualizamos el valor de theta.
end
hold on %Mantenemos la grafica anterior
plot(Mx(:,2),Mx*theta,'-'); % hacemos un plot de la 2 columna de Mx con Mx por Theta.
legend('Datos entrenamiento','Regresion Lineal');
hold off

J_theta0 = linspace(-10,10,150); % buscamos 150 valores entre -10 y 10
J_theta1 = linspace(-1,4,150); % buscamos 150 valores entre -1 y 4 

Jay = zeros(length(J_theta0),length(J_theta1)); %inicializamos una matriz Jay a ceros que luego usaremos para hacer el surface.

for i = 1:length(J_theta0)
   for j = 1:length(J_theta1)
       Mt=[J_theta0(i);J_theta1(j)];
       Jay(i,j)=(0.5/m).*(Mx*Mt-My)'*(Mx*Mt-My); % actualizamos los valores de theta para la matrix jay
   end
end

figure; % abrimos una nueva ventana de figura
surface(J_theta0,J_theta1,Jay'); % realizamos la funcion surface para graficar los valores de theta y la de la matriz Jay
hold on
plot(theta(1),theta(2),'r*');
hold off
figure; % otra ventana de figura
contour(J_theta0,J_theta1,Jay',logspace(-2,3,20));% usamos la funcion contour con 20 valores que tomamos en funcion logaritmica.
hold on
plot(theta(1),theta(2),'r*');