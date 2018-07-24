
function [cost,grad] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

                     % -------------------- YOUR CODE HERE --------------------
                     % Instructions:
                     %   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
                     %   earlier exercise onto this file, renaming the function to
                     %   sparseAutoencoderLinearCost, and changing the autoencoder to use a
                     %   linear decoder.
                     % -------------------- YOUR CODE HERE --------------------



% visibleSize: the number of input units (probably 64)
% hiddenSize: the number of hidden units (probably 25)
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example.

% The input theta is a vector (because minFunc expects the parameters to be a vector).
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
% follows the notation convention of the lecture notes.

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values).
% Here, we initialize them to zeros.
cost = 0;
W1grad = zeros(size(W1));
W2grad = zeros(size(W2));
b1grad = zeros(size(b1));
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
% forward %


Z1 = W1 * data + b1;

Hidden = sigmoid(Z1);

Z2 = W2 * Hidden + b2;

Out = Z2; % compare to sparse-ae, CHANGE here as linear output %

[p, m] = size(data);

tmp = (Out - data).^2;
J1 = sum(tmp(:))/m;

J2 = (sum(sum((W1.^2))) + sum(sum((W2.^2))));

P = ones(hiddenSize, 1);
J3 = 0;
for j = 1:hiddenSize
  sparsityParam_hat = sum(Hidden(j,:))/m;
  P(j) = sparsityParam_hat;
  J3 = J3 + sparsityParam * log(sparsityParam/sparsityParam_hat) + (1 - sparsityParam) * log((1 - sparsityParam)/(1 - sparsityParam_hat));
end

cost = J1 + J2 * lambda/2.0 + beta * J3;


%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

%---- J1

Z2grad_1 = (2.0/m) * (Out - data); % compare to sparse-ae, CHANGE here as linear backpropagation %
W2grad_1 = Z2grad_1 * Hidden';
b2grad_1 = sum(Z2grad_1,2);

Hiddengrad_1 = W2' * Z2grad_1;
Z1grad_1 = Hiddengrad_1 .* (Hidden .* (1 - Hidden));
W1grad_1 = Z1grad_1  * data';
b1grad_1 = sum(Z1grad_1, 2);

%---- J3
Hiddengrad_3 = beta * (-1*sparsityParam./P + (1 - sparsityParam)./(1 - P))/m * ones(1, m);
Z1grad_3 = Hiddengrad_3 .* (Hidden .* (1 - Hidden));
W1grad_3 = Z1grad_3 * data';
b1grad_3 = sum(Z1grad_3, 2);


%---- combine
W1grad = W1grad_1 + lambda * W1 + W1grad_3;
W2grad = W2grad_1 + lambda * W2;
b1grad = b1grad_1 + b1grad_3;
b2grad = b2grad_1;

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)).

function sigm = sigmoid(x)

    sigm = 1 ./ (1 + exp(-x));
end
