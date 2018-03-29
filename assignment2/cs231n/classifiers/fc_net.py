from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *

# 具体的两种网络结构
# 两层网络：只有最基本的 全连接层、激活
# 全连接网络：有 全连接、激活、bn、dropout
class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.
    # 一个两层全连接神经网络，使用ReLU激活和softmax loss。
    # 输入维度为D，隐藏层维度是H，分为C类
    The architecure should be affine - relu - affine - softmax.
    # f = W2*max(0,W1x)
    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.
    # 有一个单独的模块负责优化，学习参数存在self.params中
    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.
        # 只有一个隐藏层，其维度是hidden_dim，就是W1的列数，W2的行数
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        pass
        # 注意这里 w1是全连接层对应的权重，W2是隐藏层对应的权重
        self.params['W1'] = np.random.normal(0,weight_scale,(input_dim,hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(0,weight_scale,(hidden_dim,num_classes))
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.
        # 对一小部分数据，计算损失和梯度
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        pass
        ar1_out, ar1_cache = affine_relu_forward(X,self.params['W1'],self.params['b1'])
        a2_out, a2_cache = affine_forward(ar1_out,self.params['W2'],self.params['b2'])
        scores = a2_out
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
        pass
        # 拿到最后一步的loss 和 对scores求导的结果
        loss, dscores = softmax_loss(scores,y)  
        # 最终的Loss
        loss = loss + 0.5 * self.reg * ( np.sum(self.params['W1'] * self.params['W1']) + np.sum(self.params['W2'] * self.params['W2']))
        # affine全连接层 求梯度
        dx2,dw2,db2 = affine_backward(dscores, a2_cache)
        grads['W2'] = dw2 + self.reg*self.params['W2']
        grads['b2'] = db2
        # 求经过ReLU非线性层参数的梯度
        dx1,dw1,db1 = affine_relu_backward(dx2, ar1_cache)
        grads['W1'] = dw1 + self.reg*self.params['W1']
        grads['b1'] = db1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1（γ和β）; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        pass
        # 从第一层开始设置W和b
        layer_input_dim = input_dim
        for i,hd in enumerate(hidden_dims): 
            self.params['W%d'%(i+1)] = weight_scale * np.random.randn(layer_input_dim,hd)
            self.params['b%d'%(i+1)] = weight_scale * np.zeros(hd)
            if self.use_batchnorm:
                self.params['gamma%d'%(i+1)] = np.ones(hd)
                self.params['beta%d'%(i+1)] = np.zeros(hd)
            layer_input_dim = hd
        # 设置最后一层的W b
        self.params['W%d'%(self.num_layers)] = weight_scale * np.random.randn(layer_input_dim, num_classes)
        self.params['b%d'%(self.num_layers)] = weight_scale * np.random.randn(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # dropout是指在深度学习网络的训练过程中，对于神经网络单元，按照一定的概率将其暂时从网络中丢弃。对于随机梯度下降来说，由于是随机丢弃，故而每一个mini-batch都在训练不同的网络。
        # 使用dropout需要给出每一层的dropout概率 和 train/test 层 
        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed
        # bn的参数设置
        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'
        
        # dropout_param 和 bn_param 的参数都是 mode
        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # 前向传播
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
        pass
        layer_input = X
        ar_cache = {}
        dp_cache = {}
        
        for lay in range(self.num_layers-1):
            if self.use_batchnorm:
                layer_input, ar_cache[lay] = affine_bn_relu_forward(layer_input, 
                                        self.params['W%d'%(lay+1)], self.params['b%d'%(lay+1)], 
                                        self.params['gamma%d'%(lay+1)], self.params['beta%d'%(lay+1)], self.bn_params[lay])
            else:
                layer_input, ar_cache[lay] = affine_relu_forward(layer_input, self.params['W%d'%(lay+1)], self.params['b%d'%(lay+1)])
            
            if self.use_dropout:
                layer_input,  dp_cache[lay] = dropout_forward(layer_input, self.dropout_param)
        
        ar_out, ar_cache[self.num_layers] = affine_forward(layer_input, self.params['W%d'%(self.num_layers)],self.params['b%d'%(self.num_layers)])
        scores = ar_out
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # 反向传播，计算梯度（softmax + l2正则）
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        pass
        loss, dscores = softmax_loss(scores, y)
        dout = dscores
        loss += 0.5*self.reg * np.sum(self.params['W%d'%(self.num_layers)]) * np.sum(self.params['W%d'%(self.num_layers)])
        dx,dw,db = affine_backward(dout, ar_cache[self.num_layers])
        grads['W%d'%(self.num_layers)] = dw + self.reg*self.params['W%d'%(self.num_layers)]
        grads['b%d'%(self.num_layers)] = db
        dout = dx
        for idx in range(self.num_layers-1):
            # 从倒数第二层开始往回算梯度
            lay = self.num_layers -1 - idx -1
            loss = loss + 0.5*self.reg * np.sum(self.params['W%d'%(lay + 1)]) * np.sum(self.params['W%d'%(lay + 1)])
            if self.use_dropout:
                dout = dropout_backward(dout,dp_cache[lay])
            if self.use_batchnorm:
                dx,dw,db,dgamma,dbeta = affine_bn_relu_backward(dout,ar_cache[lay])
            else:
                dx,dw,db = affine_relu_backward(dout,ar_cache[lay])
            grads['W%d'%(lay+1)] = dw + self.reg*self.params['W%d'%(lay+1)]
            grads['b%d'%(lay+1)] = db
            if self.use_batchnorm:
                grads['gamma%d'%(lay+1)] = dgamma
                grads['beta%d'%(lay+1)] = dbeta
            dout = dx
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
