function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.


% forward prop
Z = theta * data; % no bias???
[max_Z, argmax_Z] = max(Z);
Z = bsxfun(@minus, Z, max(Z, [], 1)); % to prevent overflow
Out = exp(Z);
OutNorm = sum(Out);
P = Out./OutNorm; % calc probability

% logloss: -sum(sum(ylogp))/numCases

J1 = groundTruth .* log(P);
J1 = -sum(J1(:))/numCases;
J2 = lambda * sum(sum(theta.^2));
cost = J1 + J2;

% back prop
Z_grad = (P - groundTruth) /numCases;
thetagrad = thetagrad + Z_grad * data';

thetagrad = thetagrad + 2 * lambda * theta;
grad = thetagrad(:);

end
