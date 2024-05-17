import pickle, numpy as np

PATH = 'data/mnist.pkl'
IMAGE_SIZE = 784

class DataLoader:
  
  def parse_raw(self, raw_set):
    images_raw: np.ndarray
    answers_raw: np.ndarray
    images_raw, answers_raw = raw_set
    N = images_raw.shape[0]
    
    processed = []
    for i in range(N):
      answer_v = np.zeros(10)
      answer_v[answers_raw[i] - 1] = 1
      processed.append((np.reshape(images_raw[i], (IMAGE_SIZE,)) , answer_v))
    return processed
  
  def load_data(self):
    f = open(PATH, 'rb')
    training_raw, validation_raw, test_raw = pickle.load(f, encoding='bytes')
    # training data is in the format
    # (a, b) where
    # a = [<img1>, <img2>, ...] and each <imgi> is an array
    # b = [2,5,1,4,2,4,2] ... etc.
    return (self.parse_raw(training_raw), self.parse_raw(validation_raw), self.parse_raw(test_raw))
  
dl = DataLoader()
_, _, t = dl.load_data()
print(t[0:5])