function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%




m = size(X,1);

Ureduce = U(:,1:K);
for i = 1:m
  Z(i, :) = Ureduce' * X(i, :)';
endfor




end
