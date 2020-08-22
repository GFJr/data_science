function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%


cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigmas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];


% create a blank results matrix
results = zeros(length(cs) * length(sigmas), 3); 
row = 1;
for C_val = cs
  for sigma_val = sigmas
    model = svmTrain(X, y, C_val, @(x1, x2) gaussianKernel(x1, x2, sigma_val));
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));
    
    results(row,:) = [C_val sigma_val error];
    row = row + 1;
  endfor
endfor

[v i] = min(results(:,3))

C = results(i,1);
sigma = results(i,2);

end
