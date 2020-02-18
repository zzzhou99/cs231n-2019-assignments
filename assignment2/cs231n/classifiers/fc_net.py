from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.
    采用模块化设计，实现了具有ReLU非线性和softmax损失的两层全连接神经网络。
    我们假设输入维度为D，隐含维度为H，并对C类进行分类。
    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.
    注意，这个类不实现梯度下降;相反，它将与负责运行优化的独立求解器对象交互。
    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    模型的可学习参数存储在将参数名称映射到numpy数组的字典self.params中。
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights. 为权重的随机初始化提供标准差的标量。
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # 初始化两层网的权值和biases。
        # 权重应该从以0.0为中心的高斯分布初始化，其标准差等于weight_scale，biases应该初始化为0。
        # 所有的权值和偏差应该存储在字典self.params中，
        # 第一层的权值和biases使用键'W1'和'b1'，第二层的权值和biases使用键'W2'和'b2'。
        pass
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.
        如果y为空，则运行模型的测试时间前向传播，返回:
        -scores：shape数组(N, C)给出分类分数，其中scores[i, c]是X[i]和c类的分类分数。
        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        如果y不为空，则运行一个训练时间前向和反向传播，并返回一个元组:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
          具有与self.params相同键的字典，将参数名称映射到与这些参数相关的损失的梯度。
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # 实现两层网络的前向传播，计算X的类分数并将它们存储在分数变量中。
        # affine - relu - affine - softmax.
        pass
        #a1_out, a1_cache = affine_forward(X, self.params['W1'], self.params['b1'])
        #r1_out, r1_cache = relu_forward(a1_out)
        ar1_out, ar1_cache = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        a2_out, a2_cache = affine_forward(ar1_out, self.params['W2'], self.params['b2'])
        scores = a2_out
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # 实现两层网络的反向传播。将损失存储在loss变量中，梯度存储在grads字典中。
        # 使用softmax计算数据丢失，并确保grads[k]保持self.params[k]的梯度。
        # 不要忘记添加L2正则化!
        # 为了确保您的实现与我们的匹配，并通过自动化测试，
        # 请确保您的L2正则化包含一个0.5的因子，以简化梯度表达式。
        pass
#         https://zhuanlan.zhihu.com/p/20945670
#         数据损失（data loss）
        loss, dscores = softmax_loss(scores, y)
#          + 正则化损失（regularization loss）
        loss = loss + 0.5 * self.reg * (np.sum(self.params['W1'] ** 2) + 
                                        np.sum(self.params['W2'] ** 2))
        dx2, dw2, db2 = affine_backward(dscores, a2_cache)
        grads['W2'] = dw2 + self.reg * self.params['W2']
        grads['b2'] = db2
        #dx2_relu = relu_backward(dx2, r1_cache)
        #dx1, dw1, db1 = affine_backward(dx2_relu, a1_cache)
        dx1 , dw1, db1 = affine_relu_backward(dx2, ar1_cache)
        grads['W1'] = dw1 + self.reg * self.params['W1']
        grads['b1'] = db1
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be
    一个完全连接的神经网络具有任意数量的隐层，ReLU非线性和一个softmax损失函数。
    这也将实现dropout和batch/layer normalization选项。
    对于具有L层的网络，体系结构将是
    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    与上面的TwoLayerNet类似，可学习的参数存储在self.params字典中，可以使用Solver类学习。
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        给出每个隐藏层大小的整数列表。
        - input_dim: An integer giving the size of the input.
        给出输入大小的整数。
        - num_classes: An integer giving the number of classes to classify.
        给出要分类的类数的整数。
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
          在0和1之间的标量,给出了dropout强度。如果dropout=1，那么网络就不应该使用dropout。
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
          网络应该使用哪种类型的标准化。
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
          所有计算都将使用此数据类型执行。
          float32更快，但精度更低，因此应该使用float64进行数值梯度检查。
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
          如果不是None，那么通过这个随机种子来到dropout层。
          这将使dropout层是确定的，所以我们可以梯度检查模型。
        """
#         self.use_batchnorm = use_batchnorm
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # 初始化网络参数，将所有值存储在self.params字典中。
        # 在Wi和bi中存储第i层的weights和biases
        # 权重应该从一个以0为中心的正态分布初始化，其标准差等于weight_scale。
        # Biases应该初始化为零。
        # 使用batch normalization时，将第i层的scale和shift参数存储在gammai和betai中;
        # Scale参数初始化为1，shift参数初始化为0。
        pass
    
        layer_input_dim = input_dim
        for i, hd in enumerate(hidden_dims):
#             在Wi和bi中存储第i层的weights和biases
            self.params['W%d'%(i+1)] = weight_scale * np.random.randn(layer_input_dim, hd)
            self.params['b%d'%(i+1)] = weight_scale * np.zeros(hd)
            # 使用batch normalization时，将第i层的scale和shift参数存储在gammai和betai中;
#             if self.use_batchnorm:
            if self.normalization =='batchnorm':
                self.params['gamma%d'%(i+1)] = np.ones(hd)
                self.params['beta%d'%(i+1)] = np.zeros(hd)
            layer_input_dim = hd
        self.params['W%d'%(self.num_layers)] = weight_scale * np.random.randn(layer_input_dim, num_classes)
        self.params['b%d'%(self.num_layers)] = weight_scale * np.zeros(num_classes)
        
        '''
        for layer in range(self.num_layers):
#         layer_dim 第layer层的维度
            if layer==0:
                layer_dim=(input_dim,hidden_dims[layer])
            elif layer==self.num_layers-1:
                layer_dim = (hidden_dims[layer - 1], num_classes)
            else:
                layer_dim = (hidden_dims[layer - 1], hidden_dims[layer])
#             在Wi和bi中存储第i层的weights和biases
            self.params['W%d'%(layer+1)] = weight_scale * np.random.randn(layer_dim[0],layer_dim[1])
            self.params['b%d'%(layer+1)] = np.zeros(layer_dim[1])
            # 使用batch normalization时，将第i层的scale和shift参数存储在gammai和betai中;
            if self.use_batchnorm and layer!=self.num_layers-1:
                self.params['gamma%d' % (layer + 1)] = np.ones(layer_dim[1])
                self.params['beta%d' % (layer + 1)] = np.zeros(layer_dim[1])
        '''
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        # 当使用dropout时，我们需要传递一个dropout_param字典到每一个dropout层，
        # 使该层知道dropout的概率和模式(训练/测试)。
        # 您可以将相同的dropout_param传递给每个dropout层。
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        # 使用 batch normalization 时，我们需要跟踪运行均值和方差，
        # 因此需要向每个 batch normalization 层传递一个特殊的bn_param对象。
        # 将self.bn_params[0]传递给第1个batch normalization层的前向传播，
        # 将self.bn_params[1]传递给第2个batch normalization层的前向传播，等等。
        self.bn_params = []
        if self.normalization =='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization =='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        # 将所有参数转换为正确的数据类型
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        # 为batchnorm参数和dropout参数设置训练/测试模式，因为它们在训练和测试中表现不同。
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization =='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # 实现全连接网络的前向传播，计算X的类分数并将它们存储在scores变量中。
        
        # 使用dropout时，你需要传递self.dropout_param到每个dropout前向传播。
        # 当使用batch normalization时，
        # 将self.bn_params[0]传递给第1个batch normalization层的前向传播，
        # 将self.bn_params[i]传递给第i+1个batch normalization层的前向传播，等等。
        pass
        
        layer_input = X
        ar_cache = {}
        dp_cache = {}

        for lay in range(self.num_layers-1):
            # 当使用batch normalization时，
            # 将self.bn_params[0]传递给第1个batch normalization层的前向传播，
            # 将self.bn_params[i]传递给第i+1个batch normalization层的前向传播，等等。
#             if self.use_batchnorm:
            if self.normalization =='batchnorm':
                layer_input, ar_cache[lay] = affine_bn_relu_forward(layer_input, 
                                        self.params['W%d'%(lay+1)], self.params['b%d'%(lay+1)], 
    self.params['gamma%d'%(lay+1)], self.params['beta%d'%(lay+1)], self.bn_params[lay])
            else:
                layer_input, ar_cache[lay] = affine_relu_forward(layer_input, 
                                     self.params['W%d'%(lay+1)], self.params['b%d'%(lay+1)])
            # 使用dropout时，你需要传递self.dropout_param到每个dropout前向传播。
            if self.use_dropout:
                layer_input,  dp_cache[lay] = dropout_forward(layer_input, self.dropout_param)

        ar_out, ar_cache[self.num_layers] = affine_forward(layer_input, 
                   self.params['W%d'%(self.num_layers)], self.params['b%d'%(self.num_layers)])
        scores = ar_out
        
        '''
        inputi=X
        # use for BP
        fc_cache_list=[]
        relu_cache_list=[]
        bn_cache_list=[]
        dropout_cache_list=[]
        for layer in range(self.num_layers):
            #forward
            Wi,bi = self.params['W%d'%(layer+1)],
                    self.params['b%d'%(layer+1)]
            outi, fc_cachei = affine_forward(inputi, Wi, bi)
            fc_cache_list.append(fc_cachei)

            # 当使用batch normalization时，
            # 将self.bn_params[0]传递给第1个batch normalization层的前向传播，
            # 将self.bn_params[i]传递给第i+1个batch normalization层的前向传播，等等。
            #batch normalization:the last layer of the network should not be normalized
            if self.use_batchnorm and layer!=self.num_layers-1:
                gammai, betai = self.params['gamma%d' % (layer + 1)], 
                                self.params['beta%d' % (layer + 1)]
                outi, bn_cachei = batchnorm_forward(outi, gammai, betai, self.bn_params[layer])
                bn_cache_list.append(bn_cachei)
            #relu
            outi, relu_cachei = relu_forward(outi)
            relu_cache_list.append(relu_cachei)

            # 使用dropout时，你需要传递self.dropout_param到每个dropout前向传播。
            if self.use_dropout:
                outi, dropout_cachei=dropout_forward(outi, self.dropout_param)
                dropout_cache_list.append(dropout_cachei)

            inputi=outi

        scores = outi
#         '''
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # 为全连接网络实现反向传播。
        # 将损失存储在loss变量中，梯度存储在grads字典中。
        # 使用softmax计算数据损失，并确保grads[k]保持self.params[k]的梯度。
        # 不要忘记添加L2正则化!
        # 当使用batch/layer normalization时，您不需要调整缩放和移位参数。
        
        # 为了确保您的实现与我们的匹配，并通过自动化测试，
        # 请确保您的L2正则化包含一个0.5的因子，以简化梯度表达式。
        pass
    
        loss, dscores = softmax_loss(scores, y)
        dhout = dscores
        loss = loss + 0.5 * self.reg * np.sum(self.params['W%d'%(self.num_layers)] **2)
        dx , dw , db = affine_backward(dhout , ar_cache[self.num_layers])
        grads['W%d'%(self.num_layers)] = dw + self.reg * self.params['W%d'%(self.num_layers)]
        grads['b%d'%(self.num_layers)] = db
        dhout = dx
        for idx in range(self.num_layers-1):
#             反向遍历
            lay = (self.num_layers-1) - idx - 1
            loss = loss + 0.5 * self.reg * np.sum(self.params['W%d'%(lay+1)] **2)
            if self.use_dropout:
                dhout = dropout_backward(dhout ,dp_cache[lay])
#             if self.use_batchnorm:
            if self.normalization =='batchnorm':
                dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(dhout, ar_cache[lay])
            else:
                dx, dw, db = affine_relu_backward(dhout, ar_cache[lay])
            grads['W%d'%(lay+1)] = dw + self.reg * self.params['W%d'%(lay+1)]
            grads['b%d'%(lay+1)] = db
#             if self.use_batchnorm:
            if self.normalization =='batchnorm':
                grads['gamma%d'%(lay+1)] = dgamma
                grads['beta%d'%(lay+1)] = dbeta
            dhout = dx
        
        '''
        data_loss, dout = softmax_loss(scores, y)
        W_square_sum = 0
        for layer in range(self.num_layers):
            W_square_sum += (np.sum(self.params['W%d' % (layer+1)] **2))
        reg_loss = 0.5 * self.reg * W_square_sum
        loss = data_loss + reg_loss

        for layer in list(range(self.num_layers,0,-1)):
            #dropout
            if self.use_dropout:
                dout = dropout_backward(dout, dropout_cache_list[layer-1])
            #relu
            dout = relu_backward(dout, relu_cache_list[layer-1])
            #batch normalization: the last layer of the network should not be normalized
            if self.use_batchnorm and layer != self.num_layers:
                dout, dgamma, dbeta = batchnorm_backward(dout, bn_cache_list[layer-1])
                grads['gamma%d' % (layer)] = dgamma
                grads['beta%d' % (layer)] = dbeta

            #backforward
            dxi, dWi, dbi = affine_backward(dout, fc_cache_list[layer-1])
            dWi += self.reg * self.params['W%d' % (layer)]

            grads['W%d' % (layer)] = dWi
            grads['b%d' % (layer)] = dbi

            dout = np.dot(dout, self.params['W%d' % (layer)].T)
#         '''
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
