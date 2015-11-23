

function [X_norm,mu,sigma] = normalizaAtributo(X)
   if (length(X(:,1)) == 1)
       mu = X;
       sigma = X;
   else
   mu = mean(X);
   sigma = std(double(X));
   end
   
   for atri = 1:length(X(1,:))
       X_norm(:,atri) = (X(:,atri) - mu(atri))./ sigma(atri);
   end
end

