function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples






hypothesis = sigmoid(X * theta)
J = ( (1/m) * ( (-y' * log(hypothesis)) - ((1 - y)' * log(1 - hypothesis))) )

theta(1) = 0
squaredTheta = theta' * theta
regJ = ((lambda / (2*m)) * squaredTheta)
J = J + regJ


regGrad = theta * (lambda / m)
grad = ( (1/m)* (hypothesis - y)' * X )' + regGrad

end
