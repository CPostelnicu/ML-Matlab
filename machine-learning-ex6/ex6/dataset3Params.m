function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


% use the cross validation set Xval, yval to determine the best C and σ parameter to use. % You should write any additional code necessary to help you search over the parameters 
% C and σ.

% try all possible pairs of values for C and σ (e.g., C = 0.3 and σ = 0.1). 
% Selected values of C 
C_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';

% Selected values of sigma 
sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';

% if you try each of the 8 values listed above for C and for σ2, you would end up 
% training and evaluating (on the cross validation set) a total of 64 different models.

% rows associated to C, columns associated to sigma
error_val = zeros(length(C_vec), length(sigma_vec));

% 	error_val(i,j) should give 
%               the errors obtained after training with 
%               C = C_vec(i) and sigma = sigma_vec(j)
%

for indexC = 1:length(C_vec)
	for indexSigma = 1:length(sigma_vec)
	% learn theta parameters
	model= svmTrain(X, y, C_vec(indexC), @(x1, x2) gaussianKernel(x1, x2, sigma_vec(indexSigma))); 

	% evaluate the crossvalidation error on the entire cross validation set 
	predictions = svmPredict(model, Xval);
	error_val(indexC, indexSigma)  = mean(double(predictions ~= yval));
	
	end
end

%==============================================================================
% Alternative:
% [C_vec, sigma_vec, error_train, error_val] = validationCurve(X, y, Xval, yval);
%==============================================================================




% determine the best C and σ parameters to use

% error_val(i,j) errors after training with C_vec(i) and error_val(j)
% get all minimums per column (considering all sigma associated to C_vec(i,column_index))
[allcolumnmins, columnmin_index] = min(error_val);


[min_errorval, min_errorval_ind] = min(allcolumnmins);

% best sigma
sigma = sigma_vec(min_errorval_ind)

% best C
C = C_vec(columnmin_index(min_errorval_ind))









% =========================================================================

end
