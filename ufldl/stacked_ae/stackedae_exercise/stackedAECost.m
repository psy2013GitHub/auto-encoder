function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, sparsityParam, beta, data, labels)

% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.

% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example.
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example

%% setup
regularizeHidden = 0;

%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

%% forward step
dataStack = cell(numel(stack)+1,1); %每层输入
dataStack{1}.x = data;
for d = 1:numel(stack)
    % dataStack{d+1}.x = stack{d}.w * dataStack{d}.x + repmat(stack{d}.b,1,M);
    dataStack{d+1}.x = sigmoid(stack{d}.w * dataStack{d}.x + repmat(stack{d}.b,1,M));
end

Z = softmaxTheta * dataStack{numel(stack)+1}.x;
[max_Z, argmax_Z] = max(Z);
Z = bsxfun(@minus, Z, max(Z, [], 1)); % to prevent overflow
Out = exp(Z);
OutNorm = sum(Out);
POut = Out./OutNorm; % calc probability

%% cost, todo lambda是对网络所有参数还是只对最后一层，值得一提得是如果在finetune时继续对hidden层做regularize得话，效果反倒变差了。。。
J1 = groundTruth .* log(POut);
J1 = -sum(J1(:))/M;

J2 = 0;
if regularizeHidden>0
  for d = 1:numel(stack)
      J2 = J2 + lambda * sum(sum(stack{d}.w.^2));
  end
end
J2 = J2 + lambda * sum(sum(softmaxTheta.^2));

% 稀疏性仅用在隐层，不用在loss层
P = cell(numel(stack),1);
J3 = 0;
if regularizeHidden>0
  for d = 1:numel(stack)
    [currHiddenSize, ~] = size(stack{d}.w);
    hidden = dataStack{d+1}.x;
    P{d} = ones(currHiddenSize, 1);
    for j = 1:currHiddenSize
      sparsityParam_hat = sum(hidden(j,:))/M;
      P{d}(j) = sparsityParam_hat;
      J3 = J3 + sparsityParam * log(sparsityParam/sparsityParam_hat) + (1 - sparsityParam) * log((1 - sparsityParam)/(1 - sparsityParam_hat));
    end
  end
end
cost = J1 + J2 + beta * J3;

%% backward step
Z_grad = (POut - groundTruth) /M;
softmaxThetaGrad = softmaxThetaGrad + Z_grad * dataStack{numel(stack)+1}.x';
softmaxThetaGrad = softmaxThetaGrad + 2 * lambda * softmaxTheta;

aOutGrad = softmaxTheta' * Z_grad;
for d = numel(stack):-1:1
    if regularizeHidden>0
      aOutGrad = aOutGrad + beta * (-1*sparsityParam./P{d} + (1 - sparsityParam)./(1 - P{d}))/M * ones(1, M);
    end
    [currHiddenSize, ~] = size(stack{d}.w);
    gradZ = aOutGrad .* ((dataStack{d+1}.x) .* (1 - (dataStack{d+1}.x)));
    stackgrad{d}.w = gradZ * (dataStack{d}.x)';
    if regularizeHidden>0
        stackgrad{d}.w = stackgrad{d}.w + 2 * lambda * stack{d}.w;
    end
    stackgrad{d}.b = sum(gradZ, 2);
    aOutGrad = stack{d}.w' * gradZ;
end

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
