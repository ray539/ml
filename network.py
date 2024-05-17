import numpy as np

class CrossEntropyCost:
  
  @staticmethod
  def delta(aL: np.ndarray, y: np.ndarray, zL: np.ndarray):
    """
      returns the value of deltaC / delta aL * sigma'(zL)
    """
    return aL - y
  
  @staticmethod
  def fn(aL: np.ndarray, y: np.ndarray):
    """
      aL: activations on the last layer
      y:  expected output
      aL and y: 1D arrays
      
      returns cost of a single example
      note if any component in aL is 0 or 1
      it may cause the cost to be infinity
    """
    cost = 0
    N = aL.shape[0]
    for j in range(N):
      if (y[j] == 0):
        cost += (1 - y[j]) * np.log(1 - aL[j])
      elif (y[j] == 1):
        cost += y[j] * np.log(aL[j])
      else:
        cost += y[j] * np.log(aL[j]) + (1 - y[j]) * np.log(1 - aL[j])
    return cost

class Network:
  sizes: list[int]
  weights: list[np.ndarray]
  biases: list[np.ndarray]
  
  def __init__(self, sizes: list[int]) -> None:
    """
      randomnly allocate weights and biases
    """
    # eg. sizes = [728,16,16,16,10]
    self.sizes = sizes
    self.weights = [None]
    N = len(sizes)
    for i in range(N - 1):
      R = self.sizes[i + 1]
      C = self.sizes[i]
      W = np.random.randn(R, C)
      self.weights.append(W)

    self.biases = [None]
    for i in range(1, N):
      b = np.random.randn(sizes[i])
      self.biases.append(b)
  
  def feed_forward(self, x):
    """
      x: the input vector
      return the vector a, ie the activations in the last layer
    """
    activations = [x]
    L = len(self.sizes)
    for l in range(1, L):
      W = self.weights[l]
      x = activations[l - 1]
      b = self.biases[l]
      activations.append(sigmoid(W @ x + b))
    
    return activations[l - 1]
  
  def update_minibatch(self, data: list[tuple[np.ndarray, np.ndarray]], lmda, eta, N):
    """
      data: [(x1, y1), (x2, y2), ...]
      lmda: regularization parameter
      eta: learning rate
      N: length of the training data
      
      perform stochastic gradient descent
    """
    M = len(data)
    L = len(self.sizes)
    
    
    #so, get the sum over all values of nabla_b and nabla_w for all values of x over this minibatch
    sum_nBs = [None] + [np.zeros(b.shape) for b in self.biases[1:]] 
    sum_nWs = [None] + [np.zeros(w.shape) for w in self.weights[1:]]
    
    for x, y in data:
      nabla_bs, nabla_Ws = self.backprop(x, y)
      for l in range(1, L):
        sum_nBs[l] += nabla_bs[l]
        sum_nWs[l] += nabla_Ws[l]
    
    for l in range(1, L):
      self.biases[l] = self.biases[l] - (eta / M) * sum_nBs[l]
      self.weights[l] = (1 - eta * lmda / N) * self.weights[l] - (eta / M) * sum_nWs[l]
  
  
  def backprop(self, x: np.ndarray, y: np.ndarray):
    """
      x: the input data
      y: the expected output
      returns: (nabla_b, nabla_w)
      for a single training example
    """
    # first, we need to find all the activations
    activations: list[np.ndarray] = [x]
    zs = [None]
    L = len(self.sizes)
    for l in range(1, L):
      xi = activations[l - 1]
      Wi = self.weights[l]
      bi = self.biases[l]
      zi = Wi @ xi + bi
      zs.append(zi)
      activations.append(sigmoid(zi))

    # then, we need to find the values of delta via backpropagation
    deltas: list[np.ndarray] = [None for i in range(L)]
    delta_L = CrossEntropyCost.delta(activations[L - 1], y, zs[L - 1])
    deltas[L - 1] = delta_L
    l = L - 2
    while l >= 0:
      Wl1 = self.weights[l + 1]
      deltal1 = deltas[l + 1]
      zl = zs[l]
      deltal = hammard(Wl1.T @ deltal1, zl)
      deltas[l] = deltal
      l -= 1
    
    # then, we use the values of delta to get the values of nabla_W and nabla_b
    nabla_Ws: list[np.ndarray] = [None for i in range(L)]
    nabla_bs: list[np.ndarray] = [None for i in range(L)]
    l = L - 1
    # fill in L - 1 ... 1
    while L >= 1:
      nabla_bs[l] = deltas[l].copy()
      R = self.sizes[l]
      C = self.sizes[l - 1]
      nabla_W = np.ndarray((R, C))

      for j in range(R):
        for k in range(C):
          nabla_W[j][k] = deltas[l][j] * activations[l - 1][k]
      nabla_Ws[l] = nabla_W
      
      # alternate
      # nabla_W = deltas[l] @ activations[l - 1].T
      l -= 1

    return (nabla_bs, nabla_Ws)
  
def sigmoid(v: np.ndarray):
  return 1 / (1 + np.exp(v))
  
def sigmoid_prime(v: np.ndarray):
  return sigmoid(v) * (1 - sigmoid(v))

def hammard(v: np.ndarray, w: np.ndarray):
  N = v.shape[0]
  res = np.zeros(N)
  for i in range(N):
    res[i] = v[i] * w[i]
  return res