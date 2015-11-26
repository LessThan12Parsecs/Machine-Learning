function p = predecir(Theta1, Theta2, X)

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
[blabla, p] = max(h2, [], 2);

end

