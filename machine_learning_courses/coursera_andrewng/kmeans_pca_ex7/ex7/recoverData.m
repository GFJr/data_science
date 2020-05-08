function X_rec = recoverData(Z, U, K)
%RECOVERDATA Recovers an approximation of the original data when using the 
%projected data
%   X_rec = RECOVERDATA(Z, U, K) recovers an approximation the 
%   original data that has been reduced to K dimensions. It returns the
%   approximate reconstruction in X_rec.
%

Ureduce = U(:, 1:K);
for i = 1:size(Z,1)
    z = Z(i, :)';
    x_approx = Ureduce * z;
    X_rec(i, :) = x_approx';
  
endfor


end
