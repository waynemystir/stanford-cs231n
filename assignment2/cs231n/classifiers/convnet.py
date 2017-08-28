import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class CifarConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W  = input_dim
    FH, FW = filter_size, filter_size
    F = num_filters

    # conv layer
    self.params['W1'] = weight_scale * np.random.randn(F, C, FH, FW)
    self.params['b1'] = np.zeros(F)

    self.params['gamma1'] = np.ones(num_filters)
    self.params['beta1'] = np.zeros(num_filters)

    # first affine layer
    pool_height = 2
    pool_width = 2
    stride = 2

    HH = int((H - pool_height) / stride + 1)
    WW = int((W - pool_width) / stride + 1)

    self.params['W2'] = weight_scale * np.random.randn(F, F, FH, FW)
#    self.params['W2'] = weight_scale * np.random.randn(F * HH * WW, hidden_dim)
#    self.params['b2'] = np.zeros(hidden_dim)
    self.params['b2'] = np.zeros(F)

#    self.params['gamma2'] = np.ones(hidden_dim)
#    self.params['beta2'] = np.zeros(hidden_dim)

    self.params['gamma2'] = np.ones(num_filters)
    self.params['beta2'] = np.zeros(num_filters)

    # second affine layer
#    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['W3'] = weight_scale * np.random.randn(F * 16 * 16, num_classes)
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    'conv - relu - 2x2 max pool - affine - relu - affine - softmax'
    spatial_param = {'mode': 'train'}
    bn_param = {'mode': 'train'}

    gamma1, beta1 = self.params['gamma1'], self.params['beta1']
    gamma2, beta2 = self.params['gamma2'], self.params['beta2']

#    print("CCN-cff1 X({}) W1({})".format(X.shape, W1.shape))
    scores, conv1_cache = conv_forward_fast(X, W1, b1, conv_param)
    args = [scores, gamma1, beta1, spatial_param]
    scores, spatial1_cache = spatial_batchnorm_forward(*args)
    scores, relu1_cache = relu_forward(scores)
    scores, pool_cache = max_pool_forward_fast(scores, pool_param)

#    print("CCN-cff2 X({}) W2({})".format(scores.shape, W2.shape))
    scores, conv2_cache = conv_forward_fast(scores, W2, b2, conv_param)
    args2 = [scores, gamma2, beta2, spatial_param]
    scores, spatial2_cache = spatial_batchnorm_forward(*args2)
    scores, relu2_cache = relu_forward(scores)

    scores, affine1_cache = affine_forward(scores, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dscores = softmax_loss(scores, y)
    dscores, dW3, db3 = affine_backward(dscores, affine1_cache)

    dscores = relu_backward(dscores, relu2_cache)
    dscores, dgamma2, dbeta2 = spatial_batchnorm_backward(dscores, spatial2_cache)
    dscores, dW2, db2 = conv_backward_fast(dscores, conv2_cache)

    dscores = max_pool_backward_fast(dscores, pool_cache)
    dscores = relu_backward(dscores, relu1_cache)
    dscores, dgamma1, dbeta1 = spatial_batchnorm_backward(dscores, spatial1_cache)
    dscores, dW1, db1 = conv_backward_fast(dscores, conv1_cache)

    loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
    dW1 += self.reg * W1
    dW2 += self.reg * W2
    dW3 += self.reg * W3

    grads['W1'], grads['b1'] = dW1, db1
    grads['W2'], grads['b2'] = dW2, db2
    grads['W3'], grads['b3'] = dW3, db3

    grads['gamma1'], grads['beta1'] = dgamma1, dbeta1
    grads['gamma2'], grads['beta2'] = dgamma2, dbeta2
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
