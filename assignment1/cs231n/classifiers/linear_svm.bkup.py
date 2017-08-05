import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    XTi = X.T[:, i]
    scores = W.dot(XTi)
    correct_class_score = scores[y[i]]
    indicator =  (scores-correct_class_score+1)>0
    for j in xrange(num_classes):
      if j == y[i]:
        dW[j,:] += -np.sum(np.delete(indicator,j))*XTi.T
        continue
      dW[j,:] += indicator[j]*XTi.T
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  
  # Add regularization to the loss and the gradient
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss,dW


#OLD?
#def svm_loss_naive(W, X, y, reg):
#  """
#  Structured SVM loss function, naive implementation (with loops).
#
#  Inputs have dimension D, there are C classes, and we operate on minibatches
#  of N examples.
#
#  Inputs:
#  - W: A numpy array of shape (D, C) containing weights.
#  - X: A numpy array of shape (N, D) containing a minibatch of data.
#  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
#    that X[i] has label c, where 0 <= c < C.
#  - reg: (float) regularization strength
#
#  Returns a tuple of:
#  - loss as single float
#  - gradient with respect to weights W; an array of same shape as W
#  """
#  dW = np.zeros(W.shape) # initialize the gradient as zero
#
#  # compute the loss and the gradient
#  num_classes = W.shape[1]
#  num_train = X.shape[0]
#  loss = 0.0
#  for i in xrange(num_train):
#    scores = X[i].dot(W)
#    correct_class_score = scores[y[i]]
#    for j in xrange(num_classes):
#      if j == y[i]:
#        continue
#      margin = scores[j] - correct_class_score + 1 # note delta = 1
#      if margin > 0:
#        loss += margin
#
#  # Right now the loss is a sum over all training examples, but we want it
#  # to be an average instead so we divide by num_train.
#  loss /= num_train
#
#  # Add regularization to the loss.
#  loss += reg * np.sum(W * W)
#
#  #############################################################################
#  # TODO:                                                                     #
#  # Compute the gradient of the loss function and store it dW.                #
#  # Rather that first computing the loss and then computing the derivative,   #
#  # it may be simpler to compute the derivative at the same time that the     #
#  # loss is being computed. As a result you may need to modify some of the    #
#  # code above to compute the gradient.                                       #
#  #############################################################################
#
#
#  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  X = X.T
  num_train = X.shape[1]
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = W.dot(X)
  seq = np.arange(num_train)
  correct_class_scores = scores[y, seq]

  margins = np.maximum(0, scores - correct_class_scores + 1)
  margins[y, seq] = 0
  loss = np.sum(margins)

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  indicators = np.where(margins > 0, 1, 0)
  indicators[y, seq] = -1 * np.sum(indicators, 0)
  dW = indicators.dot(X.T)

  dW /= num_train
  dW += reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
