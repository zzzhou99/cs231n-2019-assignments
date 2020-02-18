from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. 
    The net has an input dimension of N, a hidden layer dimension of H, 
    and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. 
    The network uses a ReLU nonlinearity after the first fully connected layer.
    双层全连接神经网络。
    网络的输入维数为N，隐含层维数为H，对C个类进行分类。
    我们用一个softmax损失函数和L2正则化的权重矩阵来训练网络。
    在第1个全连接层之后，网络使用1个ReLU非线性。

    In other words, the network has the following architecture:
    换句话说，网络的架构如下:
    input - fully connected layer - ReLU - fully connected layer - softmax
    输入 - 全连接层 - ReLU - 全连接层 - softmax
    The outputs of the second fully-connected layer are the scores for each class.
    第2个全连通层的输出是每个类的分数。
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:
        初始化模型。
        Weights初始化为小的随机值，biases初始化为零。
        Weights和biases存储在变量self.params中。
        self.params是一个带有以下键的字典:
        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.
        计算两层全连接神经网络的损失和梯度。

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
          y[i]是X[i]的标签，每个y[i]都是 0 <= y[i] < c 范围内的整数。这个参数是可选的;
          如果它如果没有通过，那么我们只返回分数，如果通过了，那么我们返回损失和梯度。
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].
        如果y为空，则返回一个(N, C)的scores矩阵，其中scores[i, c]是C类在输入X[i]时的分数。

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples. 这批训练样本的损失(数据损失和正则化损失)。
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
          字典将参数名称映射到这些参数相对于损失函数的梯度;具有与self.params相同的键。
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # 执行前向传播，计算输入的类分数。
        # 将结果存储在scores变量中，该变量应该是一个shape (N, C)数组。
        pass
        h_output = np.maximum(0, X.dot(W1) + b1) #(N,D) * (D,H) = (N,H)
        scores = h_output.dot(W2) + b2 # (N,H) * (H, C) = (N, C)
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # 完成前向传播，计算损失。
        # 这应该包括W1和W2的数据损失和L2正则化。
        # 将结果存储在变量loss中，它应该是一个标量。
        # 使用Softmax分类器损失。
        pass
        # https://zhuanlan.zhihu.com/p/21102293
#     求 某行中的最大值 组成的行向量(1, N)，再reshape成 列向量(N, 1)
#     减去这个最大值组成的列向量，将向量f中的数值进行平移，使得最大值为0
        shift_scores = scores - np.max(scores, axis = 1).reshape(-1,1)
#     求 某行的和 组成的行向量(1, N)，再reshape成列向量(N, 1)
#     求 softmax函数输出 (N, C)
        softmax_output = np.exp(shift_scores) / np.sum(np.exp(shift_scores), axis = 1).reshape(-1,1)
#     交叉熵损失（cross-entropy loss） 
        loss = -np.sum(np.log(softmax_output[range(N), list(y)]))
        loss /= N
        loss +=  0.5* reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # 计算后向遍历，计算权重和偏差的导数。将结果存储在梯度字典中。
        # 例如，梯度['W1']应该将梯度存储在W1上，并且是相同大小的矩阵
        pass
        dscores = softmax_output.copy()
        dscores[range(N), list(y)] -= 1
        dscores /= N
        grads['W2'] = h_output.T.dot(dscores) + reg * W2
        grads['b2'] = np.sum(dscores, axis = 0)

        dh = dscores.dot(W2.T)
        dh_ReLu = (h_output > 0) * dh
        grads['W1'] = X.T.dot(dh_ReLu) + reg * W1
        grads['b1'] = np.sum(dh_ReLu, axis = 0)
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.
        利用随机梯度下降训练神经网络。

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization. 优化的学习率
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch. 在每个epoch之后用来衰减学习率的factor
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing. 
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            # 创建一个训练数据和标签的随机小批量，分别存储在X_batch和y_batch中。
            pass
            randomIndex = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[randomIndex]
            y_batch = y[randomIndex]
            
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            # 使用梯度字典中的梯度，使用随机梯度下降更新网络的参数(存储在字典self.params中)。
            # 您需要使用上面定义的梯度字典中存储的梯度。
            pass
            for param_name in self.params:
                self.params[param_name]+=-learning_rate*grads[param_name]
                
            '''
            self.params['W2'] += - learning_rate * grads['W2']
            self.params['b2'] += - learning_rate * grads['b2']
            self.params['W1'] += - learning_rate * grads['W1']
            self.params['b1'] += - learning_rate * grads['b1']
            '''
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            # 每个阶段，检查训练和val的正确率和衰减学习率。
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.
        利用这个两层网络的训练权值来预测数据点的标签。
        对于每个数据点，我们预测每个C类的得分，并将每个数据点分配给得分最高的类。
        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass
        h = np.maximum(0, X.dot(self.params['W1']) + self.params['b1'])
        scores = h.dot(self.params['W2']) + self.params['b2']
        y_pred = np.argmax(scores, axis=1)
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
