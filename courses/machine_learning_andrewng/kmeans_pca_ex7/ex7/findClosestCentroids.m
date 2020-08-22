function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);
m = size(X,1);
% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

distance_matrix = zeros(m, K);

for j = 1:K
  d = X - centroids(j,:);
  d = (abs(d)).^2;
  sum_matrix = sum(d,2);
  distance_matrix(:,j) = sum_matrix';
endfor

[minval, idx] = min(distance_matrix, [], 2);


end

