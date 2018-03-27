import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    # y[i]是第i个图像的正确分类值，scores[y[i]]是正确分类的分数
    correct_class_score = scores[y[i]]
    # 开始计算SVM损失
    for j in xrange(num_classes):
        # 如果是正确的类别，那个分数就不要，不正确的分类，如果差距>1,再累加到loss中
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
    # 下面是根据公式计算出来的分类正确和错误时的梯度
      if margin > 0:
        loss += margin
        dW[:,j] += X[i]  # 分错误
        dW[:,y[i]] -= X[i]  # 分正确
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.loss是每个图片loss加和的平均值
  loss /= num_train
  dW /= num_train
    
  # Add regularization to the loss.再加上L2正则化项，*0.5是因为求导的时候好算。L2正则求导 = reg*W,所以L的最终梯度 = 如下
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


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  pass
  N = X.shape[0]
  scores = X.dot(W)  # N*C,N是图片数目，C是分类数,range(0,N)会创建一个list,包括0不包括N，range(N)也是一样生成0-（N-1）的矩阵
  print (scores.shape)
  margin = scores - scores[range(0,N),y].reshape(-1,1) + 1  # N*C,全部分数都 - 每一行的正确类别的分数 + 1 ，# y中的元素要在  scores列的角标范围内，即类别也应该是 0 - 9，
  margin[range(N), y] = 0 # 正确的那一类都是0，不算在损失内
  margin = (margin > 0) * margin # 将margin中大于0的元素保留下来,小于0的归为0
  loss += margin.sum() / N  # svm loss求和，求平均值
  loss += 0.5 * reg * np.sum(W * W)  # 再加上l2正则
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
  pass
  counts = (margin > 0).astype(int) # 将margin中>0的元素保留为1
  counts[range(N), y] = - np.sum(counts, axis = 1) # np.sum是沿着横轴把所有数加起来，就是有多少个数没分对
  dW += np.dot(X.T, counts) / N + reg * W   # X.T是X的转置
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
