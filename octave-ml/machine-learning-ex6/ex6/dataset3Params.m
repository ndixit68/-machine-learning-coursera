function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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

C_samples = sigma_samples =[0.01,0.03,0.1,0.3,1,3,10,30];
Choice_matrix =[]

for i = 1:numel(C_samples)
  for j = 1:numel(sigma_samples)
    model = svmTrain(X, y, C_samples(1,i), @(x1, x2) gaussianKernel(x1, x2, sigma_samples(1,j)));
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));
    Choice_matrix= [Choice_matrix;[C_samples(1,i) sigma_samples(1,j) error]];
  end
end

[min_val, index] = min(Choice_matrix(:,3),[],1);

C = Choice_matrix(index, 1);
sigma = Choice_matrix(index, 2);

% =========================================================================

end
