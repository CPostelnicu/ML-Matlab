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


% recode the output labels as num_labels vectors containing only values 0 or 1
I = eye(num_labels);
yy = zeros(num_labels, length(y));
for index = 1:length(y)
 yy(:, index) = I(y(index),:)';
end

%feedforward neural network
a_1= [ones(m,1) X]; % add a0_1

% hidden layer
z_2 = Theta1 * a_1'; % X contains examples in rows. Theta1 contains the parameters in rows.
a_2= sigmoid(z_2);

[ma2, na2] = size(a_2);
a_2 = [ones(1, na2); a_2]'; % add a0_2

z_3= a_2 * Theta2';
a_3 = sigmoid(z_3);


% J = J -(1/m)*  sum(sum(yy'.*log(a_3) + (1-yy)'.*log(1-a_3)));

J = J -(1/m)*  sum(sum(yy'.*log(a_3) + (1-yy)'.*log(1-a_3)))+ (lambda/(2*m)) * sum(sum (Theta1(:,2:end).^2)) + (lambda/(2*m)) * sum(sum (Theta2(:,2:end).^2));


%====================================================================================
% BACKPROPAGATION ALGORITHM
% loop that processes one example at a time. 
% index-th iteration performing the calculation on the index-th training example (x(index),y(index)).
%====================================================================================
Dlt_1 = zeros(size(Theta1));
Dlt_2 = zeros(size(Theta2));
for index = 1:m

%====================================================================================
% Step 1: run a “forward pass” to compute all the activations throughout the network, including the 
% output value of the hypothesis hΘ(x). 
% Perform a feedforward pass, computing the activations z_2, a_2, z_3, a_3 for layers 2 and 3. 
% Note: must add a +1 term to ensure that the vectors of activations for layers a_1 and a_2 also 
% include the bias unit. 	 
%====================================================================================
 
	a_1= [1; X(index,:)']; % add a0_1. a_1 becomes a 401 x 1 column vector.
	
	
	% The parameters have dimensions that are sized for a neural network with 
	% 25 units in the second layer and 
	% 10 output units (corresponding to the 10 digit classes).
	
	% hidden layer
	z_2 = Theta1 * a_1; % Theta1 contains the parameters in rows (25 x 401). a_1 is a 401 x 1 column vector.
	a_2= sigmoid(z_2);
	a_2 = [1; a_2]; % add a0_2 . a_2 becomes a 26 x 1 column vector
	
	
	z_3= Theta2 * a_2; % Theta2 contains the parameters in rows (10 x 26). a_2 is a 26 x 1 column vector
	a_3 = sigmoid(z_3); % a_3 becomes a 10 x 1 column vector. 

	
%====================================================================================
% Step 2: for each node j in layer l, compute an “error term” δ(l) that measures how much that node 
% was  “responsible” j for any errors in our output.
% For each output unit k in layer 3 (the output layer), 
% set δk(3) = (ak(3) − yk),
%% where yk ∈ {0,1} indicates whether the current training example belongs to class k (yk = 1), 
% or if it belongs to a different class (yk = 0).
% (“backpropagate the errors from output layer to layer 2”)
%====================================================================================
	
	% for each output unit k in layer 3 (the output layer)
	delta_3 = a_3 - yy(:,index); % yy a 10 x 5000 vector. delta_3 a 10 x 1 column vector.
	
%====================================================================================
% Step 3: For the hidden layer l = 2, set% δ(2) = 􏰀Θ(2)􏰁' δ(3). ∗ g′(z(2))
%====================================================================================
	delta_2 = Theta2(:,2:end)' * delta_3 .* sigmoidGradient(z_2);

%====================================================================================
% Step 4: Accumulate the gradient from this example: 
% ∆(l) = ∆(l) + δ(l+1)(a(l))T
% Note: you should skip or remove δ0(2). 
% This corresponds to corresponds to delta 2 = delta 2(2:end). 
%====================================================================================
	Dlt_2 = Dlt_2 + delta_3 * a_2';
	Dlt_1 = Dlt_1 + delta_2 * a_1';

end

% Obtain the (unregularized) gradient for the neural network cost function by dividing the 
% accumulated gradients by 1/m.
% Return the partial derivatives of the cost function with respect to Theta1 and Theta2 in 
% Theta1_grad and Theta2_grad.


        %Theta1_grad = Dlt_1/m;
	%Theta2_grad = Dlt_2/m;

	
	Theta1_grad(:,1) = Dlt_1(:,1)/m;
	Theta2_grad(:,1) = Dlt_2(:,1)/m;
	
	Theta1_grad(:, 2:end) = Dlt_1(:,2:end)/m + (lambda/m) * Theta1(:,2:end);
	Theta2_grad(:, 2:end) = Dlt_2(:,2:end)/m + (lambda/m) * Theta2(:,2:end);
	
	
	%Theta1_grad =[Dlt_1/m (Dlt_1(:,2:end)/m + (lambda/m) * Theta1(:,2:end))];
	%Theta2_grad = [Dlt_2/m (Dlt_2(:,2:end)/m + (lambda/m)* Theta2(:,2:end))];



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
