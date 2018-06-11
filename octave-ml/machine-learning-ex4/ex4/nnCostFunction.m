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
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
A1 = [ones(m, 1) X];
Z2 = A1 * Theta1';
A2 = sigmoid(Z2);
%

% adding bias unit in layer 1
A2 = [ones(m,1) A2];
Z3 = A2*Theta2';
%sigmoidal below represents A3 as A3 is the last layer, i.e output layer
%_htheta = 1./(1+(e.^-Z3));
h_theta = sigmoid(Z3);

% convert y into m X K array
new_y = zeros(m,num_labels);

for i=1:size(y,1)
    new_y(i,y(i,:))= 1.0;
end

cost = sum(new_y.*log(h_theta)+(1-new_y).*log(1-h_theta),2);

theta1ExcludingBias = Theta1(:,2:end);
theta2ExcludingBias = Theta2(:,2:end);
reg1 = sum(sum(theta1ExcludingBias.^2));
reg2 = sum(sum(theta2ExcludingBias.^2));

regularization_term = lambda*(reg1+reg2)/(2*m);

J=-sum(cost)/m +regularization_term;

% -------------------------------------------------------------
% -----------------------BACK PROPAGATION----------------------
% -------------------------------------------------------------

delta2 = zeros(size(Theta1));
delta3 = zeros(size(Theta2));

delta3 = h_theta - new_y;
%delta2 = Theta2'*delta2.*sigmoidGradient(Z2);

delta2 = (Theta2'*delta3')'.*(A2.*(1-A2));

% =========================================================================
capitaldelta1 = zeros(size(Theta1));
capitaldelta2 = zeros(size(Theta2));

for i=1:m
  % For the input layer, where l=1:
  X_ith = X(i,:);
  A1_ith = [1 X_ith];
  
  % For the hidden layers, where l=2:
  Z2_ith = A1_ith * Theta1';
  A2_ith = sigmoid(Z2_ith);
  A2_ith = [1 A2_ith];

  % For the output layer, where l=3:
  Z3_ith = A2_ith * Theta2';
  A3_ith = sigmoid(Z3_ith);

  % For the delta values:
  delta3_ith = A3_ith - new_y(i,:);
  delta2_ith = (Theta2'*delta3_ith'.*sigmoidGradient([1 Z2_ith]'))';
  delta2_ith = delta2_ith(2:end); %Taking of the bias row

  capitaldelta2 = capitaldelta2 + delta3_ith' * A2_ith;
  capitaldelta1 = capitaldelta1 + delta2_ith' * A1_ith;

end
 
%Theta1_grad = (1/m)*capitaldelta1;   
%Theta2_grad = (1/m)*capitaldelta2;

 
Theta1ZeroedBias = [ zeros(size(Theta1, 1), 1) theta1ExcludingBias ];
Theta2ZeroedBias = [ zeros(size(Theta2, 1), 1) theta2ExcludingBias ];
Theta1_grad = (1 / m) * capitaldelta1 + (lambda / m) * Theta1ZeroedBias;
Theta2_grad = (1 / m) * capitaldelta2 + (lambda / m) * Theta2ZeroedBias;

 % Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
