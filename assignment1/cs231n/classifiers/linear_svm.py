from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).
    结构化的SVM损失函数，naive实现（带循环）。
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
    输入的维数为D，有C类，并且我们对N个示例的小批量进行操作。

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; 
         y[i] = c means that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    返回一个元组：
      单精度浮点损失
      关于权重W的梯度； 与W形状相同的数组
    """
    # initialize the gradient as zero
    dW = np.zeros(W.shape)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W) # (1, C) = (1, D)*(D, C)
        correct_class_score = scores[y[i]]
#         https://zhuanlan.zhihu.com/p/20945670
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin

                # 计算 j不=yi 时的行的梯度
                dW[:, j] += X[i].T # (D, 1) = (1, D).T
                # 计算 j=yi 时的行的梯度
                dW[:, y[i]] += -X[i].T # (D, 1) = (1, D).T

#      https://zhuanlan.zhihu.com/p/20945670      
    # 计算数据损失（data loss）
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
#     现在损失是所有训练例子的总和，但是我们希望它是一个平均值，所以我们除以num_train。
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    # 加上正则化损失（regularization loss）
    loss += reg * np.sum(W * W)
    dW += reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # 计算损失函数的梯度并将其存储为dW。
    # 与其先计算损失再计算导数，还不如在计算损失的同时计算导数更简单。
    # 因此，您可能需要修改上面的一些代码来计算梯度。
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_classes = W.shape[1]
    num_train = X.shape[0]

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # 实现一个结构化支持向量机损失的向量化版本，储存结果到 loss
    # https://zhuanlan.zhihu.com/p/20945670   

#     每一行是一个样本的C个类别的分值
    scores = X.dot(W) # (N, C) = (N, D)*(D, C)
#     利用表示了第i个训练样本的正确类别的y[i]来找到每一个训练样本的正确分类的分值
# 弄成一个列向量 correct_class_scores
    correct_class_scores = scores[range(num_train), list(y)].reshape(-1, 1)  # (N, 1)
    margin = np.maximum(0, scores - correct_class_scores + 1) # (N, C)
#     j不=yj 所以把正确的分类的 置为0
    margin[range(num_train), list(y)] = 0
#     2个求和 所以直接sum
    loss += np.sum(margin) / num_train # 数据损失（data loss）
    loss += 0.5 * reg * np.sum(W * W) # 正则化损失（regularization loss）
    
    '''
    Loss = W.dot(X) - (W.dot(X))[y,np.arange(num_train)]+1
    # On sum sur les colonnes et on enlceve la valeur que l'on a compteur en trop
    Loss = np.sum(Loss * (Loss > 0) , axis = 0) - 1.0
    Regularization = 0.5 * reg * np.sum(W*W)
    loss = np.sum(Loss) / float(num_train) +Regularization
    '''
    '''
    scores = X.dot(W) # (N, C) = (N, D)*(D, C)
    # 500*10的矩阵,表示500个image的ground truth
    correct_class_score = scores[np.arange(num_train), y]
    # 重复10次,得到500*10的矩阵,才可以和scores相加相减
    correct_class_score = np.reshape(np.repeat(correct_class_score, num_classes), 
                                     (num_train, num_classes))
    margin = scores - correct_class_score + 1.0
    margin[np.arange(num_train), y] = 0
    loss = (np.sum(margin[margin > 0])) / num_train # 数据损失（data loss）
    loss += 0.5 * reg * np.sum(W * W) # 正则化损失（regularization loss）
    '''
    '''
    scores = np.dot(X, W) # (N, C) = (N, D)*(D, C)
    margin = scores - scores[range(0, num_train), y].reshape(num_train, 1) + 1
    margin[range(0, num_train), y] = 0
    margin = margin * (margin > 0)  # max(0, s_j - s_yi + delta)
    loss += np.sum(margin) / num_train # 数据损失（data loss）
    loss += 0.5 * reg * np.sum(W * W) # 正则化损失（regularization loss）
    '''
    '''
    scores = X.dot(W)   # (N, C) = (N, D)*(D, C)
    margin = scores - scores[range(0, num_train), y].reshape(-1, 1) + 1  # N x C
    margin[range(num_train), y] = 0
    margin = (margin > 0) * margin  # max(0, s_j - s_yi + delta)
    loss += margin.sum() / num_train # 数据损失（data loss）
    loss += 0.5 * reg * np.sum(W * W) # 正则化损失（regularization loss）
    '''
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
# 为结构化的支持向量机损失实现一个向量化的梯度，并将结果存储在dW中。
# 与从头开始计算梯度不同，重用一些用于计算损失的中间值可能更容易。

    margin[margin > 0] = 1  # (N, C)
    margin[margin <= 0] = 0
    margin[np.arange(num_train), y] = -np.sum(margin, axis=1) # (N, 1).T
    # for xi in range(num_train):
    #   dW+=np.reshape(X[xi],(dW.shape[0],1))*\
    #       np.reshape(margin[xi],(1,dW.shape[1]))
    dW += np.dot(X.T, margin) / num_train + reg * W  # D by C (N, D)

    '''
    counts = (margin > 0).astype(int)
    counts[range(num_train), y] = - np.sum(counts, axis=1)
    dW += np.dot(X.T, counts) / num_train + reg * W
    '''
    '''
    coeff_mat = np.zeros((num_train, num_classes))
    coeff_mat[margins > 0] = 1
    # coeff_mat[range(num_train), list(y)] = 0
    coeff_mat[range(num_train), list(y)] = -np.sum(coeff_mat, axis=1)
    dW = (X.T).dot(coeff_mat) / num_train + reg * W
    '''
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
