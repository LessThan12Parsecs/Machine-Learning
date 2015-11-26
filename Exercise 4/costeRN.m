function [J,grad] = costeRN(params_rn,num_entradas,num_ocultas,...
    num_etiquetas,X,y,lambda)
    %costeRN calcula el coste y el gradiente de una red neuronal de dos
    %capas
    
    Theta1 = reshape(params_rn(1:num_ocultas*(num_entradas+1)),... 
      num_ocultas,(num_entradas+1));
    Theta2 = reshape(params_rn((1+(num_ocultas*(num_entradas+1))):end),... 
      num_etiquetas,(num_ocultas+1));
   
  %el numero de ejemplos de entrenamiento
  m = size(X,1);
  J = 0;
  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));
% yAux = eye(num_etiquetas);
% y = yAux(y,:);
a1 = [ones(m, 1) X];

z2 = a1 * Theta1';
a2 = sigmoide(z2);
a2 = [ones(size(a2,1), 1) a2];

z3 = a2 * Theta2';
a3 = sigmoide(z3);
hThetaX = a3;

yVec = zeros(m,num_etiquetas);

for i = 1:m
    yVec(i,y(i)) = 1;
end

% for i = 1:m
%     
%     term1 = -yVec(i,:) .* log(hThetaX(i,:));
%     term2 = (ones(1,num_labels) - yVec(i,:)) .* log(ones(1,num_labels) - hThetaX(i,:));
%     J = J + sum(term1 - term2);
%     
% end
% 
% J = J / m;

J = 1/m * sum(sum(-1 * yVec .* log(hThetaX)-(1-yVec) .* log(1-hThetaX)));

regularator = (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))) * (lambda/(2*m));

J = J + regularator;

% Part 2 implementation



for t = 1:m

	% For the input layer, where l=1:
	a1 = [1; X(t,:)'];

	% For the hidden layers, where l=2:
	z2 = Theta1 * a1;
	a2 = [1; sigmoide(z2)];

	z3 = Theta2 * a2;
	a3 = sigmoide(z3);

	yy = ([1:num_etiquetas]==y(t))';
	% For the delta values:
	delta_3 = a3 - yy;

	delta_2 = (Theta2' * delta_3) .* [1; sigmoideGradiente(z2)];
	delta_2 = delta_2(2:end); % Taking of the bias row

	% delta_1 is not calculated because we do not associate error with the input    

	% Big delta update
	Theta1_grad = Theta1_grad + delta_2 * a1';
	Theta2_grad = Theta2_grad + delta_3 * a2';
end

Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end