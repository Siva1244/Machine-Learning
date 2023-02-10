function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
X = [ones(m, 1) X];
th1=size(Theta1,1);
th2=size(Theta2,1);

for i=1:1:m
  L2=zeros(th1,1);
  L3=zeros(num_labels,1);
  for j=1:1:th1
    L2(j,1)=sigmoid(Theta1(j,:)*X(i,:)');
  endfor
  L2=[1;L2];
  for j=1:1:num_labels
    L3(j,1)=sigmoid(Theta2(j,:)*L2);
  endfor
  [m,o]=max(L3);
  p(i,1)=o;
endfor






% =========================================================================


end
