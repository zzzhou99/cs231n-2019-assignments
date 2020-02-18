from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
from cs231n.classifiers.linear_svm import *
from cs231n.classifiers.softmax import *
from past.builtins import xrange


class LinearClassifier(object):

    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.
        使用随机梯度下降训练这个线性分类器。
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.每个维度D有N个训练样本。
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.优化学习率。
        - reg: (float) regularization strength.正则化强度。
        - num_iters: (integer) number of steps to take when optimizing 优化时要采取的步骤数
        - batch_size: (integer) number of training examples to use at each step.
        在每个步骤中使用的训练示例的数量。
        - verbose: (boolean) If true, print progress during optimization.
        如果为真，则在优化期间打印进度。
        Outputs: 包含在每个训练迭代中损失函数值的列表。
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape
        # 假设y取0...K-1，其中K是类的数量
        # assume y takes values 0...K-1 where K is number of classes
        num_classes = np.max(y) + 1 
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

#         运行随机梯度下降法优化W
        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            # 从训练数据中抽取batch_size元素及其对应的标签，在这一轮梯度下降中使用。
            # 将数据存储在X_batch中，相应的标签存储在y_batch中;
            # 采样后，X_batch应该有shape (batch_size, dim)， 
            # y_batch应该有shape (batch_size, dim)
            # 使用np.random.choice来生成索引。
            # 带替换的抽样比不带替换的抽样快。
            pass
        
            batch_idx = np.random.choice(num_train, batch_size, replace = True)
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            '''
            randomIndex = np.random.choice(len(X),batch_size,replace=True)
            X_batch = X[randomIndex]
            y_batch = y[randomIndex]
            '''
            '''
            indices = np.random.choice(num_train,batch_size,replace = True)
            X_batch = X[:,indices]
            y_batch = y[indices]
            '''
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update 执行参数更新
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            # 使用梯度和学习率更新权重。
            pass
            self.W += - learning_rate * grad
            
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.
        利用该线性分类器训练好的权值来预测数据点的标签。
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.每个维度D有N个训练样本。

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
          X中数据的预测标签。 
          y_pred是一个长度为N的一维数组，每个元素都是一个整数，表示所预测的类。
        """
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # 实现这个方法。将预测的标签存储在y_pred中。
        pass
        scores = X.dot(self.W)
#         y_pred是scores每一行的最大值的索引组成的行向量，代表每个N的预测标签
        y_pred = np.argmax(scores, axis = 1)
          
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.计算损失函数及其导数。
        Subclasses will override this.子类将覆盖它

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float 单精度浮点的loss
        - gradient with respect to self.W; an array of the same shape as W
        """
        pass


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """
    # 使用多类SVM损失函数的子类

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """
    # 使用Softmax +交叉熵损失函数的子类

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
