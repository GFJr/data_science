function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);


%Matrix (5000 x 10) containing the labels
y_matrix = eye(num_labels)(y,:);


a1 = [ones(m, 1) X];;

  
#PART 1 FP
z2 = a1 * Theta1';

a2= sigmoid(z2);

m2 = size(a2, 1);
a2 = [ones(m2,1) a2];
z3 = a2 * Theta2' ;
%Matrix (1 x 10) containing the predictions
hypothesis = sigmoid(z3);
  
  
%PART 2 : BP

delta3 = hypothesis - y_matrix;


  
delta2 = delta3 * (Theta2(:,2:end)) .* sigmoidGradient(z2);


Delta1 = delta2' * a1;


Delta2 = delta3' * a2;


#COST computation and regularization
J = ((1/m) * (sum(sum(( (-y_matrix .* log(hypothesis)) - ((1 - y_matrix) .* log(1 - hypothesis)))))));

Theta1(:,1) = 0;
squaredTheta1 = Theta1 .* Theta1;
double_sum_left = sum(sum(squaredTheta1));

Theta2(:,1) = 0;
squaredTheta2 = Theta2 .* Theta2;
double_sum_right = sum(sum(squaredTheta2));

regularization_term = ((lambda / (2*m)) * (double_sum_left + double_sum_right));
J = J + regularization_term;



#Gradient computation with regularization
Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2;

Theta1(:,1) = 0;
Theta2(:,1) = 0;

Theta1_regularization = Theta1 * (lambda/m);
Theta2_regularization = Theta2 * (lambda/m);

Theta1_grad = (1/m) * Delta1 + Theta1_regularization ;
Theta2_grad = (1/m) * Delta2 + Theta2_regularization ;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
