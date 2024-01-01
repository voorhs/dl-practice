import numpy as np


def NLL(activations, target):
  """
  Returns negative log-likelihood of target under model represented by
  activations (log probabilities of classes, it's just output of LogSoftmax layer).
  Input shapes: (batch, num_classes), (batch,)
  Output shape: 1 (scalar).
  """
  return np.choose(target, -1 * activations.T).mean()


def grad_NLL(activations, target):
  """
  Returns gradient of negative log-likelihood w.r.t. activations.
  each arg has shape (batch, num_classes)
  output shape: (batch, num_classes)
  """
  res = np.zeros_like(activations)
  for i in range(res.shape[0]):
    res[i, target[i]] = -1
  return res / res.shape[0]