function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h=[];k=0;
for i=1:1:m
  h(i)= sigmoid(theta'*X(i,:)');
  J=J+(-y(i)*log( h(i) )-(1-y(i))*log(1- h(i) ));
endfor
for j=2:1:size(theta,1)
  k=k+ (theta(j,1)^2);
endfor
k=k*(lambda/2);
J=(J+k)/m;


% gradient calculation
for j=1:1:size(theta,1)
  gra=0;
   for i=1:1:m
   gra=gra+((h(i)-y(i))*X(i,j));
 endfor
 if j==1
 grad(j,1)=(1/m)*gra;
else
 grad(j,1)=(1/m)*(gra + lambda*theta(j,1));
 endif
endfor





% =============================================================

end
