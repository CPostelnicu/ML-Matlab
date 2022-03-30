function [C_vec, sigma_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select C and sigma
%   [C_vec, sigma_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of C and sigma. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of C 
C_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';

% Selected values of sigma 
sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';

% you should try all possible pairs of values for C and σ (e.g., C = 0.3 and σ = 0.1). 
% For  example, if you try each of the 8 values listed above for C and for σ2, you would end up 
% training and evaluating (on the cross validation set) a total of 82 = 64 different models.

% You need to return these variables correctly.
error_train = zeros(length(C_vec), length(sigma_vec));
error_val = zeros(length(C_vec), length(sigma_vec));

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. 
%               error_train(i,j), and error_val(i,j) should give 
%               the errors obtained after training with 
%               C = C_vec(i) and sigma = sigma_vec(j)
%


 for indexC = 1:length(C_vec)
	for indexSigma = 1:length(sigma_vec)

	% learn theta parameters
	model= svmTrain(X, y, C_vec(indexC), @(x1, x2) gaussianKernel(x1, x2, sigma_vec(indexSigma))); 

	error_train(indexC, indexSigma) = svmCostFunction(model, X, y, C_vec(indexC), sigma);

	% evaluate the crossvalidation error on the entire cross validation set 
	error_val(indexC, indexSigma) = svmCostFunction(model, Xval, yval, C_vec(indexC), sigma);

	end
end



% =========================================================================

end
