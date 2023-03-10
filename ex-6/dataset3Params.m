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
Cval=[0.01 0.03 0.1 0.3 1 3 10 30];
sigmaval=[0.01 0.03 0.1 0.3 1 3 10 30];
%error=zeros(size(Cval),size(sigmaval));
er=1;
for i=1:1:size(Cval,2)
  C=Cval(1,i);
  for j=1:1:size(sigmaval,2)
    sigma=sigmaval(1,j);
    x1 = [1 2 1]; x2 = [0 4 -1];
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
    predictions = svmPredict(model, Xval);
    error=mean(double(predictions ~= yval));
    if (abs(error)<er)
      er=abs(error);
      a=i;
      b=j;
    endif
 endfor
endfor

C=Cval(1,a);
sigma=sigmaval(1,b);



% =========================================================================

end
