function p = predecir(Theta1, Theta2, X)
    m = size(X, 1);
    num_etiquietas = size(Theta2, 1);
    h1 = sigmoide([ones(m, 1) X] * Theta1');
    h2 = sigmoide([ones(m, 1) h1] * Theta2');
    [blabla, p] = max(h2, [], 2);

end

