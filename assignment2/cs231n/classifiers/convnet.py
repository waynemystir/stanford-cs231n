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

class CifarConvNet_02(object):
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
    self.params['b2'] = np.zeros(F)

    self.params['W3'] = weight_scale * np.random.randn(F, F, FH, FW)
    self.params['b3'] = np.zeros(F)

    self.params['gamma2'] = np.ones(num_filters)
    self.params['gamma3'] = np.ones(num_filters)
    self.params['beta2'] = np.zeros(num_filters)
    self.params['beta3'] = np.zeros(num_filters)

    # second affine layer
#    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['W4'] = weight_scale * np.random.randn(F * 16 * 16, num_classes)
    self.params['b4'] = np.zeros(num_classes)
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
    W4, b4 = self.params['W4'], self.params['b4']
    
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
    gamma3, beta3 = self.params['gamma3'], self.params['beta3']

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
 
    scores, conv3_cache = conv_forward_fast(scores, W3, b3, conv_param)
    args3 = [scores, gamma3, beta3, spatial_param]
    scores, spatial3_cache = spatial_batchnorm_forward(*args3)
    scores, relu3_cache = relu_forward(scores)

    scores, affine1_cache = affine_forward(scores, W4, b4)
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
    dscores, dW4, db4 = affine_backward(dscores, affine1_cache)

    dscores = relu_backward(dscores, relu3_cache)
    dscores, dgamma3, dbeta3 = spatial_batchnorm_backward(dscores, spatial3_cache)
    dscores, dW3, db3 = conv_backward_fast(dscores, conv3_cache)

    dscores = relu_backward(dscores, relu2_cache)
    dscores, dgamma2, dbeta2 = spatial_batchnorm_backward(dscores, spatial2_cache)
    dscores, dW2, db2 = conv_backward_fast(dscores, conv2_cache)

    dscores = max_pool_backward_fast(dscores, pool_cache)
    dscores = relu_backward(dscores, relu1_cache)
    dscores, dgamma1, dbeta1 = spatial_batchnorm_backward(dscores, spatial1_cache)
    dscores, dW1, db1 = conv_backward_fast(dscores, conv1_cache)

    loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2) + np.sum(W4**2))
    dW1 += self.reg * W1
    dW2 += self.reg * W2
    dW3 += self.reg * W3
    dW4 += self.reg * W4

    grads['W1'], grads['b1'] = dW1, db1
    grads['W2'], grads['b2'] = dW2, db2
    grads['W3'], grads['b3'] = dW3, db3
    grads['W4'], grads['b4'] = dW4, db4

    grads['gamma1'], grads['beta1'] = dgamma1, dbeta1
    grads['gamma2'], grads['beta2'] = dgamma2, dbeta2
    grads['gamma3'], grads['beta3'] = dgamma3, dbeta3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass

class CifarConvNet_03(object):
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
    self.params['b2'] = np.zeros(F)

    self.params['W3'] = weight_scale * np.random.randn(F, F, FH, FW)
    self.params['b3'] = np.zeros(F)

    self.params['gamma2'] = np.ones(num_filters)
    self.params['gamma3'] = np.ones(num_filters)
    self.params['beta2'] = np.zeros(num_filters)
    self.params['beta3'] = np.zeros(num_filters)

    # second affine layer
#    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['W4'] = weight_scale * np.random.randn(F * 16 * 4, num_classes)
    self.params['b4'] = np.zeros(num_classes)
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
    W4, b4 = self.params['W4'], self.params['b4']
    
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
    gamma3, beta3 = self.params['gamma3'], self.params['beta3']

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
    scores, pool2_cache = max_pool_forward_fast(scores, pool_param)
 
    scores, conv3_cache = conv_forward_fast(scores, W3, b3, conv_param)
    args3 = [scores, gamma3, beta3, spatial_param]
    scores, spatial3_cache = spatial_batchnorm_forward(*args3)
    scores, relu3_cache = relu_forward(scores)

    scores, affine1_cache = affine_forward(scores, W4, b4)
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
    dscores, dW4, db4 = affine_backward(dscores, affine1_cache)

    dscores = relu_backward(dscores, relu3_cache)
    dscores, dgamma3, dbeta3 = spatial_batchnorm_backward(dscores, spatial3_cache)
    dscores, dW3, db3 = conv_backward_fast(dscores, conv3_cache)

    dscores = max_pool_backward_fast(dscores, pool2_cache)
    dscores = relu_backward(dscores, relu2_cache)
    dscores, dgamma2, dbeta2 = spatial_batchnorm_backward(dscores, spatial2_cache)
    dscores, dW2, db2 = conv_backward_fast(dscores, conv2_cache)

    dscores = max_pool_backward_fast(dscores, pool_cache)
    dscores = relu_backward(dscores, relu1_cache)
    dscores, dgamma1, dbeta1 = spatial_batchnorm_backward(dscores, spatial1_cache)
    dscores, dW1, db1 = conv_backward_fast(dscores, conv1_cache)

    loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2) + np.sum(W4**2))
    dW1 += self.reg * W1
    dW2 += self.reg * W2
    dW3 += self.reg * W3
    dW4 += self.reg * W4

    grads['W1'], grads['b1'] = dW1, db1
    grads['W2'], grads['b2'] = dW2, db2
    grads['W3'], grads['b3'] = dW3, db3
    grads['W4'], grads['b4'] = dW4, db4

    grads['gamma1'], grads['beta1'] = dgamma1, dbeta1
    grads['gamma2'], grads['beta2'] = dgamma2, dbeta2
    grads['gamma3'], grads['beta3'] = dgamma3, dbeta3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass

class CifarConvNet_04(object):
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
    self.params['b2'] = np.zeros(F)

    self.params['W3'] = weight_scale * np.random.randn(F, F, FH, FW)
    self.params['b3'] = np.zeros(F)

    self.params['gamma2'] = np.ones(num_filters)
    self.params['gamma3'] = np.ones(num_filters)
    self.params['beta2'] = np.zeros(num_filters)
    self.params['beta3'] = np.zeros(num_filters)

    # second affine layer
#    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['W4'] = weight_scale * np.random.randn(F * 16, num_classes)
    self.params['b4'] = np.zeros(num_classes)
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
    W4, b4 = self.params['W4'], self.params['b4']
    
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
    gamma3, beta3 = self.params['gamma3'], self.params['beta3']

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
    scores, pool2_cache = max_pool_forward_fast(scores, pool_param)
 
    scores, conv3_cache = conv_forward_fast(scores, W3, b3, conv_param)
    args3 = [scores, gamma3, beta3, spatial_param]
    scores, spatial3_cache = spatial_batchnorm_forward(*args3)
    scores, relu3_cache = relu_forward(scores)
    scores, pool3_cache = max_pool_forward_fast(scores, pool_param)

    scores, affine1_cache = affine_forward(scores, W4, b4)
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
    dscores, dW4, db4 = affine_backward(dscores, affine1_cache)

    dscores = max_pool_backward_fast(dscores, pool3_cache)
    dscores = relu_backward(dscores, relu3_cache)
    dscores, dgamma3, dbeta3 = spatial_batchnorm_backward(dscores, spatial3_cache)
    dscores, dW3, db3 = conv_backward_fast(dscores, conv3_cache)

    dscores = max_pool_backward_fast(dscores, pool2_cache)
    dscores = relu_backward(dscores, relu2_cache)
    dscores, dgamma2, dbeta2 = spatial_batchnorm_backward(dscores, spatial2_cache)
    dscores, dW2, db2 = conv_backward_fast(dscores, conv2_cache)

    dscores = max_pool_backward_fast(dscores, pool_cache)
    dscores = relu_backward(dscores, relu1_cache)
    dscores, dgamma1, dbeta1 = spatial_batchnorm_backward(dscores, spatial1_cache)
    dscores, dW1, db1 = conv_backward_fast(dscores, conv1_cache)

    loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2) + np.sum(W4**2))
    dW1 += self.reg * W1
    dW2 += self.reg * W2
    dW3 += self.reg * W3
    dW4 += self.reg * W4

    grads['W1'], grads['b1'] = dW1, db1
    grads['W2'], grads['b2'] = dW2, db2
    grads['W3'], grads['b3'] = dW3, db3
    grads['W4'], grads['b4'] = dW4, db4

    grads['gamma1'], grads['beta1'] = dgamma1, dbeta1
    grads['gamma2'], grads['beta2'] = dgamma2, dbeta2
    grads['gamma3'], grads['beta3'] = dgamma3, dbeta3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass

class CifarConvNet_05(object):
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
    self.params['b2'] = np.zeros(F)

    self.params['W3'] = weight_scale * np.random.randn(F, F, FH, FW)
    self.params['b3'] = np.zeros(F)

    self.params['gamma2'] = np.ones(num_filters)
    self.params['gamma3'] = np.ones(num_filters)
    self.params['beta2'] = np.zeros(num_filters)
    self.params['beta3'] = np.zeros(num_filters)

    # second affine layer
#    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['W4'] = weight_scale * np.random.randn(F * 16, num_classes)
    self.params['b4'] = np.zeros(num_classes)
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
    W4, b4 = self.params['W4'], self.params['b4']
    
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
    gamma3, beta3 = self.params['gamma3'], self.params['beta3']

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
    scores, pool2_cache = max_pool_forward_fast(scores, pool_param)
 
    scores, conv3_cache = conv_forward_fast(scores, W3, b3, conv_param)
    args3 = [scores, gamma3, beta3, spatial_param]
    scores, spatial3_cache = spatial_batchnorm_forward(*args3)
    scores, relu3_cache = relu_forward(scores)
    scores, pool3_cache = max_pool_forward_fast(scores, pool_param)

    scores, affine1_cache = affine_forward(scores, W4, b4)
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
    dscores, dW4, db4 = affine_backward(dscores, affine1_cache)

    dscores = max_pool_backward_fast(dscores, pool3_cache)
    dscores = relu_backward(dscores, relu3_cache)
    dscores, dgamma3, dbeta3 = spatial_batchnorm_backward(dscores, spatial3_cache)
    dscores, dW3, db3 = conv_backward_fast(dscores, conv3_cache)

    dscores = max_pool_backward_fast(dscores, pool2_cache)
    dscores = relu_backward(dscores, relu2_cache)
    dscores, dgamma2, dbeta2 = spatial_batchnorm_backward(dscores, spatial2_cache)
    dscores, dW2, db2 = conv_backward_fast(dscores, conv2_cache)

    dscores = max_pool_backward_fast(dscores, pool_cache)
    dscores = relu_backward(dscores, relu1_cache)
    dscores, dgamma1, dbeta1 = spatial_batchnorm_backward(dscores, spatial1_cache)
    dscores, dW1, db1 = conv_backward_fast(dscores, conv1_cache)

    loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2) + np.sum(W4**2))
    dW1 += self.reg * W1
    dW2 += self.reg * W2
    dW3 += self.reg * W3
    dW4 += self.reg * W4

    grads['W1'], grads['b1'] = dW1, db1
    grads['W2'], grads['b2'] = dW2, db2
    grads['W3'], grads['b3'] = dW3, db3
    grads['W4'], grads['b4'] = dW4, db4

    grads['gamma1'], grads['beta1'] = dgamma1, dbeta1
    grads['gamma2'], grads['beta2'] = dgamma2, dbeta2
    grads['gamma3'], grads['beta3'] = dgamma3, dbeta3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass

class CifarConvNet_06(object):
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
    self.params['b2'] = np.zeros(F)

    self.params['W3'] = weight_scale * np.random.randn(F, F, FH, FW)
    self.params['b3'] = np.zeros(F)

    self.params['W4'] = weight_scale * np.random.randn(F, F, FH, FW)
    self.params['b4'] = np.zeros(F)

    self.params['gamma2'] = np.ones(num_filters)
    self.params['gamma3'] = np.ones(num_filters)
    self.params['gamma4'] = np.ones(num_filters)
    self.params['beta2'] = np.zeros(num_filters)
    self.params['beta3'] = np.zeros(num_filters)
    self.params['beta4'] = np.zeros(num_filters)

    # second affine layer
#    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['W5'] = weight_scale * np.random.randn(F * 16, num_classes)
    self.params['b5'] = np.zeros(num_classes)
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
    W4, b4 = self.params['W4'], self.params['b4']
    W5, b5 = self.params['W5'], self.params['b5']
    
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
    gamma3, beta3 = self.params['gamma3'], self.params['beta3']
    gamma4, beta4 = self.params['gamma4'], self.params['beta4']

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
    scores, pool2_cache = max_pool_forward_fast(scores, pool_param)
 
    scores, conv3_cache = conv_forward_fast(scores, W3, b3, conv_param)
    args3 = [scores, gamma3, beta3, spatial_param]
    scores, spatial3_cache = spatial_batchnorm_forward(*args3)
    scores, relu3_cache = relu_forward(scores)
    scores, pool3_cache = max_pool_forward_fast(scores, pool_param)
 
    scores, conv4_cache = conv_forward_fast(scores, W4, b4, conv_param)
    args4 = [scores, gamma4, beta4, spatial_param]
    scores, spatial4_cache = spatial_batchnorm_forward(*args4)
    scores, relu4_cache = relu_forward(scores)

    scores, affine1_cache = affine_forward(scores, W5, b5)
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
    dscores, dW5, db5 = affine_backward(dscores, affine1_cache)

    dscores = relu_backward(dscores, relu4_cache)
    dscores, dgamma4, dbeta4 = spatial_batchnorm_backward(dscores, spatial4_cache)
    dscores, dW4, db4 = conv_backward_fast(dscores, conv4_cache)

    dscores = max_pool_backward_fast(dscores, pool3_cache)
    dscores = relu_backward(dscores, relu3_cache)
    dscores, dgamma3, dbeta3 = spatial_batchnorm_backward(dscores, spatial3_cache)
    dscores, dW3, db3 = conv_backward_fast(dscores, conv3_cache)

    dscores = max_pool_backward_fast(dscores, pool2_cache)
    dscores = relu_backward(dscores, relu2_cache)
    dscores, dgamma2, dbeta2 = spatial_batchnorm_backward(dscores, spatial2_cache)
    dscores, dW2, db2 = conv_backward_fast(dscores, conv2_cache)

    dscores = max_pool_backward_fast(dscores, pool_cache)
    dscores = relu_backward(dscores, relu1_cache)
    dscores, dgamma1, dbeta1 = spatial_batchnorm_backward(dscores, spatial1_cache)
    dscores, dW1, db1 = conv_backward_fast(dscores, conv1_cache)

    loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2) + np.sum(W4**2) + np.sum(W5**2))
    dW1 += self.reg * W1
    dW2 += self.reg * W2
    dW3 += self.reg * W3
    dW4 += self.reg * W4
    dW5 += self.reg * W5

    grads['W1'], grads['b1'] = dW1, db1
    grads['W2'], grads['b2'] = dW2, db2
    grads['W3'], grads['b3'] = dW3, db3
    grads['W4'], grads['b4'] = dW4, db4
    grads['W5'], grads['b5'] = dW5, db5

    grads['gamma1'], grads['beta1'] = dgamma1, dbeta1
    grads['gamma2'], grads['beta2'] = dgamma2, dbeta2
    grads['gamma3'], grads['beta3'] = dgamma3, dbeta3
    grads['gamma4'], grads['beta4'] = dgamma4, dbeta4
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass


class CifarConvNet_07(object):
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
    self.params['b2'] = np.zeros(F)

    self.params['W3'] = weight_scale * np.random.randn(F, F, FH, FW)
    self.params['b3'] = np.zeros(F)

    self.params['W4'] = weight_scale * np.random.randn(F, F, FH, FW)
    self.params['b4'] = np.zeros(F)

    self.params['W5'] = weight_scale * np.random.randn(F, F, FH, FW)
    self.params['b5'] = np.zeros(F)

    self.params['gamma2'] = np.ones(num_filters)
    self.params['gamma3'] = np.ones(num_filters)
    self.params['gamma4'] = np.ones(num_filters)
    self.params['gamma5'] = np.ones(num_filters)
    self.params['beta2'] = np.zeros(num_filters)
    self.params['beta3'] = np.zeros(num_filters)
    self.params['beta4'] = np.zeros(num_filters)
    self.params['beta5'] = np.zeros(num_filters)

    # second affine layer
#    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['W6'] = weight_scale * np.random.randn(F * 16 * 4, num_classes)
    self.params['b6'] = np.zeros(num_classes)
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
    W4, b4 = self.params['W4'], self.params['b4']
    W5, b5 = self.params['W5'], self.params['b5']
    W6, b6 = self.params['W6'], self.params['b6']
    
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
    gamma3, beta3 = self.params['gamma3'], self.params['beta3']
    gamma4, beta4 = self.params['gamma4'], self.params['beta4']
    gamma5, beta5 = self.params['gamma5'], self.params['beta5']

#    print("CCN-cff1 X({}) W1({})".format(X.shape, W1.shape))
    scores, conv1_cache = conv_forward_fast(X, W1, b1, conv_param)
    args = [scores, gamma1, beta1, spatial_param]
    scores, spatial1_cache = spatial_batchnorm_forward(*args)
    scores, relu1_cache = relu_forward(scores)
#    scores, pool_cache = max_pool_forward_fast(scores, pool_param)

#    print("CCN-cff2 X({}) W2({})".format(scores.shape, W2.shape))
    scores, conv2_cache = conv_forward_fast(scores, W2, b2, conv_param)
    args2 = [scores, gamma2, beta2, spatial_param]
    scores, spatial2_cache = spatial_batchnorm_forward(*args2)
    scores, relu2_cache = relu_forward(scores)
#    scores, pool2_cache = max_pool_forward_fast(scores, pool_param)
 
    scores, conv3_cache = conv_forward_fast(scores, W3, b3, conv_param)
    args3 = [scores, gamma3, beta3, spatial_param]
    scores, spatial3_cache = spatial_batchnorm_forward(*args3)
    scores, relu3_cache = relu_forward(scores)
    scores, pool3_cache = max_pool_forward_fast(scores, pool_param)
 
    scores, conv4_cache = conv_forward_fast(scores, W4, b4, conv_param)
    args4 = [scores, gamma4, beta4, spatial_param]
    scores, spatial4_cache = spatial_batchnorm_forward(*args4)
    scores, relu4_cache = relu_forward(scores)
    scores, pool4_cache = max_pool_forward_fast(scores, pool_param)
 
    scores, conv5_cache = conv_forward_fast(scores, W5, b5, conv_param)
    args5 = [scores, gamma5, beta5, spatial_param]
    scores, spatial5_cache = spatial_batchnorm_forward(*args5)
    scores, relu5_cache = relu_forward(scores)

    scores, affine1_cache = affine_forward(scores, W6, b6)
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
    dscores, dW6, db6 = affine_backward(dscores, affine1_cache)

    dscores = relu_backward(dscores, relu5_cache)
    dscores, dgamma5, dbeta5 = spatial_batchnorm_backward(dscores, spatial5_cache)
    dscores, dW5, db5 = conv_backward_fast(dscores, conv5_cache)

    dscores = max_pool_backward_fast(dscores, pool4_cache)
    dscores = relu_backward(dscores, relu4_cache)
    dscores, dgamma4, dbeta4 = spatial_batchnorm_backward(dscores, spatial4_cache)
    dscores, dW4, db4 = conv_backward_fast(dscores, conv4_cache)

    dscores = max_pool_backward_fast(dscores, pool3_cache)
    dscores = relu_backward(dscores, relu3_cache)
    dscores, dgamma3, dbeta3 = spatial_batchnorm_backward(dscores, spatial3_cache)
    dscores, dW3, db3 = conv_backward_fast(dscores, conv3_cache)

#    dscores = max_pool_backward_fast(dscores, pool2_cache)
    dscores = relu_backward(dscores, relu2_cache)
    dscores, dgamma2, dbeta2 = spatial_batchnorm_backward(dscores, spatial2_cache)
    dscores, dW2, db2 = conv_backward_fast(dscores, conv2_cache)

#    dscores = max_pool_backward_fast(dscores, pool_cache)
    dscores = relu_backward(dscores, relu1_cache)
    dscores, dgamma1, dbeta1 = spatial_batchnorm_backward(dscores, spatial1_cache)
    dscores, dW1, db1 = conv_backward_fast(dscores, conv1_cache)

    loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2) + np.sum(W4**2) + np.sum(W5**2) + np.sum(W6**2))
    dW1 += self.reg * W1
    dW2 += self.reg * W2
    dW3 += self.reg * W3
    dW4 += self.reg * W4
    dW5 += self.reg * W5
    dW6 += self.reg * W6

    grads['W1'], grads['b1'] = dW1, db1
    grads['W2'], grads['b2'] = dW2, db2
    grads['W3'], grads['b3'] = dW3, db3
    grads['W4'], grads['b4'] = dW4, db4
    grads['W5'], grads['b5'] = dW5, db5
    grads['W6'], grads['b6'] = dW6, db6

    grads['gamma1'], grads['beta1'] = dgamma1, dbeta1
    grads['gamma2'], grads['beta2'] = dgamma2, dbeta2
    grads['gamma3'], grads['beta3'] = dgamma3, dbeta3
    grads['gamma4'], grads['beta4'] = dgamma4, dbeta4
    grads['gamma5'], grads['beta5'] = dgamma5, dbeta5
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass

