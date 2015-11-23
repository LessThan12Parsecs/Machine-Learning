%Practica 1 Aprendizaje Automatico
%Emanuel Ramirez Catapano
%Regresion Lineal, Varias Variables.
%Cargando datos de archivo ex1data2.txt

fileID = fopen('ex1data2.txt'); % Abrimos el archivo y lo asignamos a un ID.
C  = textscan(fileID,'%f, %f, %f'); % cargamos los datos en un cell C.
fclose(fileID); % cerramos el archivo.


m = length(C{3});
M = [C{1},C{2},C{3}]; %convertimos el Cell C a una matriz M

x_anormal = [ones(size(M(:,1))), M(:,1:2)]; % guardamos los valores sin normalizar para usarlos luego con la ecuacion normal.
y = M(:,3);
[M_norm,mu,sigma] = normalizaAtributo(M(:,1:2)); % normalizamos los datos
x = [ones(size(M(:,1))), M_norm];

alpha = [0.3,0.1,0.03,0.01,0.003,0.001]; % valores de alpha que vamos a probar.
NUM_IT = 100; % numero de iteraciones para cada alpha

%creamos un cell con distintos estilos de plot para usarlos luegos en cada
%grafica de convergencia.
figure;
plotstyle = {'r', 'g', 'b', 'y', 'r--', 'g--'};

for ap = 1:length(alpha)
    theta = zeros(size(x(1,:)))'; % creamos un vector theta con ceros. 
    Jay = zeros(NUM_IT,1);
    for iter=1:NUM_IT
        %Calculamos el J
        Jay(iter) = (0.5/m) .* (x * theta - y)' * (x * theta - y);
        %El gradiente
        grad = (1/m) .* x' * ((x * theta) - y);
        %Actualizamos theta
        theta=theta-alpha(ap).*grad;
    end
    %hacemos la graficas de las diferentes alphas.
    plot(0:NUM_IT-1, Jay(1:NUM_IT), char(plotstyle(ap)),'LineWidth',3)
    hold on
    
    %En la grafica se ve que de los valores elegidos para alpha el 0.3 es
    %el que converge mas rapidamente, por lo tanto lo guardamos como
    %nuestro alpha optimo.
    if(alpha(ap) == 0.3)
        thetaGrad = theta;
    end
end

legend('0.3','0.1','0.03', '0.01', '0.003', '0.001');


%realizamos una estimacion con el theta del descenso de gradiente para una
%casa de 1600 pies y 3 habitaciones.
casa = [1650,3];
estim_grad=dot(thetaGrad,[1,(1650-mu(1))/sigma(1),(3-mu(2))/sigma(2)])

%Calculamos el theta con el metodo de ecuacion normal.
theta_n = (x_anormal' * x_anormal)\x_anormal' * y;
%y estimamos con los mismos valores 
estim_n = dot(theta_n,[1,casa])