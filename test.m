a3 = 6;
y  = 7;
theta2 = [1,1];
z2     = [2,1];
delta3 = a3 - y;

delta2 = theta2 * delta3 .* derivative_sigmoid(z2);