function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

hypothesis = X * theta;


J = (1/(2*m)) .* (sum((hypothesis - y).^2));

theta(1) = 0;
squaredTheta = theta' * theta;
regJ = ((lambda / (2*m)) * squaredTheta);
J = J + regJ;


regGrad = theta * (lambda / m);
grad = ( (1/m)* (hypothesis - y)' * X )' + regGrad;

%grad = grad(:);

end
