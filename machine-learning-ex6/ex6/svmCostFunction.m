function [J] = svmCostFunction(theta, X, y, C, sigma)
%LRCOSTFUNCTION Compute cost sum
%   J = LRCOSTFUNCTION(theta, X, y, C, sigma) computes the cost of using
%   theta as the parameter for SVM  

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

J = C* sum(y'*log(sigmoid(X*theta)) + (1-y)'*log(1-sigmoid(X*theta))) + (1/2) * sum (theta(2:end).^2);





% =============================================================

grad = grad(:);

end
