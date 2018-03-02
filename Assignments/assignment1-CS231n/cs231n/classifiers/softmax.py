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
  classes = W.shape[1]
  train = X.shape[0]

  for i in range(train):
    f = np.dot(X[i], W)
    f -= np.max(f)  # to avoid potential blowup
    p = np.exp(f) / np.sum(np.exp(f))

    local_loss = -1.0 * np.log(p[y[i]])  # neg loss
    loss += local_loss

    for j in range(classes):
        dW[:, j] += ((p[j]) - (j == y[i])) * X[i]

  loss = loss / train + reg * np.sum(W * W)
  dW = dW / train + 2 * reg * np.sum(W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  # regularization!                                                               #
  #############################################################################
  classes = W.shape[1]
  train = X.shape[0]

  f = np.dot(X, W)
  f -= np.max(f, axis=1, keepdims=True)

  p = np.exp(f) / np.sum(np.exp(f), axis=1, keepdims=True)
#   print(p[np.arange(train), y])
  loss = np.sum(-np.log(p[np.arange(train), y]))

  ind = np.zeros_like(p)
  ind[np.arange(train), y] = 1
  dW = X.T.dot(p - ind)

  loss = loss / train + reg * np.sum(W * W)
  dW = dW / train + 2 * reg * np.sum(W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

