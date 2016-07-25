function D = derivative_sigmoid(x)
D = sigmoid(x) .* (1 - sigmoid(x));
