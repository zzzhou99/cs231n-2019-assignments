import numpy as np

"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:
该文件实现各种一阶更新规则，这些规则通常用于训练神经网络。
每个更新规则接受当前权值和相对于这些权值的损失梯度，并生成下一组权值。
每个更新规则都有相同的接口:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.
    包含超参数值如学习速率、动量等的字典。
    如果更新规则需要在多次迭代中缓存值，那么config也将保存这些缓存的值。

Returns:
  - next_w: The next point after the update. 更新后的下一点。
  - config: The config dictionary to be passed to the next iteration of the
    update rule. 要传递给更新规则的下一个迭代的配置字典。

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.
对于大多数更新规则，默认的学习率可能不会执行得很好;
但是，其他超参数的默认值应该可以很好地处理各种不同的问题。
For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
为了提高效率，更新规则可以执行就地更新，修改w并将next_w设置为w。
"""


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.
    执行普通的随机梯度下降。

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
      给出动量值的标量（0~1）。设置 momentum = 0 时，退化为随机梯度下降
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
      一个与w和dw形状相同的numpy数组，用来存储梯度的移动平均值。
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the momentum update formula. Store the updated value in #
    # the next_w variable. You should also use and update the velocity v.     #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # 执行动量更新公式。
    # 将更新后的值存储在next_w变量中。
    # 你应该也使用和更新速度v。
    pass
    v = config['momentum'] * v - config['learning_rate'] * dw
    next_w = w + v
        
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    config['velocity'] = v

    return next_w, config



def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.
    使用RMSProp更新规则，该规则使用平方梯度值的滑动平均值来设置每个参数的自适应学习速率。

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients. 梯度的二阶矩的滑动平均。
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of w #
    # in the next_w variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # 实现RMSprop更新公式，将w的下一个值存储在next_w变量中。
    # 不要忘记更新保存在config['cache']中的缓存值。
    pass
    config['cache'] = config['decay_rate'] * config['cache'] + (1 - config['decay_rate']) * (dw**2)
    next_w = w - config['learning_rate'] * dw / (np.sqrt(config['cache']) + config['epsilon'])
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.
    使用Adam更新规则，该规则包含了梯度及其平方的滑动平均和bias校正项。

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(w))
    config.setdefault('v', np.zeros_like(w))
    config.setdefault('t', 0)

    next_w = None
    ###########################################################################
    # TODO: Implement the Adam update formula, storing the next value of w in #
    # the next_w variable. Don't forget to update the m, v, and t variables   #
    # stored in config.                                                       #
    #                                                                         #
    # NOTE: In order to match the reference output, please modify t _before_  #
    # using it in any calculations.                                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # 实现Adam更新公式，将w的下一个值存储在next_w变量中。
    # 不要忘记更新配置中存储的m、v和t变量。
    # 为了匹配参考输出，请在任何计算中使用t之前修改它。
    pass
    beta1, beta2, eps = config['beta1'], config['beta2'], config['epsilon']
    t, m, v = config['t'], config['m'], config['v']
    m = beta1 * m + (1 - beta1) * dw
    v = beta2 * v + (1 - beta2) * (dw * dw)
    t += 1
    alpha = config['learning_rate'] * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
    w -= alpha * (m / (np.sqrt(v) + eps))
    config['t'] = t
    config['m'] = m
    config['v'] = v
    next_w = w
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config
