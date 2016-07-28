% data generation
train_data  = [0,0,1;0,1,0;1,0,0;1,1,1];
test_data   = [0,0,1;0,1,0;1,0,0;1,1,1];

% neural network configuration

% number of nodes
num_nodes1  = 3;
num_nodes2  = 3;
num_nodes3  = 1;

% 1. input layer

% prepend ones to the training data as bias units
size_train = size( train_data );
one = ones(size_train(1),1);
input_layer = [one, train_data];
for i=1:size_train(1)
% thetas
theta1      = -20 + 40*rand(num_nodes1,num_nodes2)
theta2      = -20 + 40*rand(num_nodes2,num_nodes3)


% remove the labels of the training data
input_layer(:,size_train(2)+1) = []; 

% 2. hidden layer
hidden_layer = zeros(num_nodes2,1);
hidden_layer(1) = 1;
hidden_layer = sigmoid(input_layer * theta1);

% prepend ones to the hidden layer as bias units
% hidden_layer = [one, hidden_layer];

% 3.output layer
 output_layer = sigmoid(hidden_layer* theta2)

% 4.cost function
cost = 0;
for i = 1:size_train(1)
    label   = train_data(i, size_train(2));
    h       = output_layer(i);
cost = cost + (-1/size_train(1)) *  (label * log(h) + (1 - label) * log(1-h)) ;
end
% foreward propagation!!!!!!!!!!
a1 = test_data;
z2 = a1 * theta1;
a2 = sigmoid(z2);
z3 = a2 * theta2;
a3 = sigmoid(z3);

 
% back propagation!!!!!!!!!!!
delta3 = a3 - label;
delta2 = delta3 * transpose(theta2).* derivative_sigmoid(z2);

% visulization lambda
lambda = 3;
% delta  = transpose(a2) * delta3 + transpose(a1) * delta2;
delta3_2 = transpose(a2) * delta3;
delta2_1 = transpose(a1) * delta2;
delta    = delta + sum(delta2_1(:)) + sum(delta3_2(:)) + ;


end