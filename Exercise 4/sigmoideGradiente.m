
function [g] = sigmoideGradiente(z)
    f = 1.0 ./ (1.0 + exp( -z ));
    g = f .* (ones(size(f)) - f);
end

