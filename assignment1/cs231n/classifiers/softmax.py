from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # 使用显式循环计算软softmax损失及其梯度。
    # 将损失存储在loss中，梯度存储在dW中。
    # 如果在这里不小心，很容易遇到数值不稳定性。
    # 不要忘记正规化!
#     https://zhuanlan.zhihu.com/p/21102293

    num_classes = W.shape[1]
    num_train = X.shape[0]
    # Softmax分类器 使用 交叉熵损失（cross-entropy loss）（即 softmax 函数的负对数）
#     遍历第i个训练样本
    for i in range(num_train):
    # for i in xrange(num_train):
        # 计算 分类评分向量f (1, C)
        f = np.dot(X[i], W)  # (1, C) = (1, D)*(D, C)
        # f = X[i].dot(W)  # (1, C) = (1, D)*(D, C)
        
        # 将向量f中的数值进行平移，使得最大值为0
        f -= np.max(f)  # f.shape = num_train
        # f -= max(f)
        
        # 交叉熵损失（cross-entropy loss） (1, C)
        loss = loss + np.log(np.sum(np.exp(f))) - f[y[i]]
        # loss = loss + np.log(sum(np.exp(f))) - f[y[i]]

#         https://blog.csdn.net/zt_1995/article/details/62227603
        for j in xrange(num_classes):
#         softmax函数 输出
            softmax_output = np.exp(f[j]) / sum(np.exp(f))
            if j == y[i]:
                dW[:, j] += (-1 + softmax_output) * X[i]
            else:
                dW[:, j] += softmax_output * X[i]
                
        '''
        dW[:, y[i]] -= X[i]
        s = np.exp(f).sum()
        for j in range(num_train):
            dW[:, j] += np.exp(f[j]) / s * X[i]
        '''

    '''
    scores = X.dot(W)
    maxLogC = np.max(scores, axis=1)
    maxLogC = np.reshape(np.repeat(maxLogC, num_classes), scores.shape)
    # 将向量f中的数值进行平移，使得最大值为0
    expScores = np.exp(scores + maxLogC)
    for i in range(num_train):
        # substract maxnium to make the exp standard
        esum = sum(expScores[i])
        eyi = expScores[i, y[i]]
        li = -np.log(eyi / esum)
        loss += li
        for j in range(num_classes):
            dW[:, j] += (expScores[i, j] / esum) * X[i]
        dW[:, y[i]] -= X[i]
    '''
    # 整个数据集的损失值 = 数据集中所有样本数据的损失值Li的均值 + 正则化损失R(W)
    loss = loss / num_train + 0.5 * reg * np.sum(W * W)
    dW = dW / num_train + reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]

    scores = X.dot(W)  # scores.shape = num_train, num_classes  = (N, D) * (D, C)
#     求 某行中的最大值 组成的行向量(1, N)，再reshape成 列向量(N, 1)
#     减去这个最大值组成的列向量，将向量f中的数值进行平移，使得最大值为0
    scores -= np.max(scores, axis=1).reshape(-1, 1)
#     求 某行的和 组成的行向量(1, N)，再reshape成列向量(N, 1)
#     求 softmax函数输出 (N, C)
    softmax_output = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(-1, 1)
#     交叉熵损失（cross-entropy loss） 
    loss = -np.sum(np.log(softmax_output[range(num_train), list(y)]))
#     https://blog.csdn.net/zt_1995/article/details/62227603
    dS = softmax_output.copy()
    dS[range(num_train), list(y)] += -1
    dW = (X.T).dot(dS)
    
    '''
    f = np.dot(X, W)  # f.shape = num_train, num_classes  = (N, D) * (D, C)
#     求 某行中的最大值 组成的行向量(1, N)，再reshape成 列向量(N, 1)
#     减去这个最大值组成的列向量，将向量f中的数值进行平移，使得最大值为0
    f -= f.max(axis=1).reshape(num_train, 1)
#     求 某行的和 组成的行向量(1, N)
    s = np.exp(f).sum(axis=1)
#     交叉熵损失（cross-entropy loss） 
    loss = np.log(s).sum() - f[range(num_train), y].sum()
#     https://blog.csdn.net/zt_1995/article/details/62227603
#     求 某行的和 组成的行向量(1, N)，再reshape成列向量(N, 1)
#     求 softmax函数输出 (N, C)
    counts = np.exp(f) / s.reshape(num_train, 1)
    counts[range(num_train), y] -= 1
    dW = np.dot(X.T, counts)
    '''
    '''
    scores = X.dot(W)
    maxLogC = np.max(scores, axis=1)
    maxLogC = np.reshape(np.repeat(maxLogC, num_classes), scores.shape)
    expScores = np.exp(scores + maxLogC)
    exp_correct_class_score = expScores[np.arange(num_train), y]
    # loss
    loss = -np.log(exp_correct_class_score / np.sum(expScores, axis=1))
    loss = sum(loss)
#     https://blog.csdn.net/zt_1995/article/details/62227603
    # gradient
    expScoresSumRow = np.reshape(np.repeat(np.sum(expScores, axis=1), num_classes), expScores.shape)
    graidentMatrix = expScores / expScoresSumRow
    # 对于yi要-1
    graidentMatrix[np.arange(num_train), y] -= 1
    dW = X.T.dot(graidentMatrix)
    # dW[np.arange(num_classes),y] -= X[y,]
    '''
    
    loss = loss / num_train + 0.5 * reg * np.sum(W * W)
    dW = dW / num_train + reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
