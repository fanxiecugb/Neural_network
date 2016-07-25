% data generation
train_data  = [0,0,1;0,1,0;1,0,0;1,1,1];
test_data   = [0,0,1;0,1,0;1,0,0;1,1,1];

% neural network configuration

% number of nodes
num_nodes1  = 3;
num_nodes2  = 3;
num_nodes3  = 1;

% thetas
theta1      = -20 + 40*rand(num_nodes1,num_nodes2 - 1)
theta2      = -20 + 40*rand(num_nodes2,num_nodes3)



% 1. input layer

% prepend ones to the training data as bias units
size_train = size( train_data );
one = ones(size_train(1),1);
input_layer = [one, train_data];

% remove the labels of the training data
input_layer(:,size_train(2)+1) = []; 

% 2. hidden layer
hidden_layer = zeros(num_nodes2,1);
hidden_layer(1) = 1;
hidden_layer = sigmoid(input_layer * theta1);

% prepend ones to the hidden layer as bias units
hidden_layer = [one, hidden_layer];

% 3.output layer
output_layer = sigmoid(hidden_layer* theta2)

% 4.cost function
cost = 0;
for i = 1:size_train(1)
    label   = train_data(i, size_train(2));
    h       = output_layer(i);
cost = cost + (-1/size_train(1)) *  (label * log(h) + (1 - label) * log(1-h)) ;
end
% foreward propergation!!!!!!!!!!

% decision boundary
a = zeros(size_train(1),1);

for i = 1:size_train(1)
   if output_layer(i) >= 0.5
       a(i) = 1;
   else
       a(i) = 0;
   end
   % back propagation!!!!!!!!!!!
   
end






