from src.api.save_data import save_data_http
from src.helper import url_solver as US
from src.constants import jobs as J


def init_dataset_mnist_http(url_data_server):
  from tensorflow.keras.datasets.mnist import load_data
  (x_train, y_train), (x_test, y_test) = load_data()

  #Describimos los datos X
  norm_x_train = x_train.astype('float32') / 255
  norm_x_test = x_test.astype('float32') / 255

  # One hot encoding is a representation of categorical variables as binary vectors. https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
  from tensorflow.keras.utils import to_categorical
  encoded_y_train = to_categorical(y_train, num_classes=10, dtype='float32')
  encoded_y_test = to_categorical(y_test, num_classes=10, dtype='float32')

  X_train = norm_x_train.reshape(-1, 28, 28, 1)
  Y_train = encoded_y_train
  X_test = norm_x_test.reshape(-1, 28, 28, 1)
  Y_test = encoded_y_test

  # save numpy array as npy file
  import numpy as np
  from numpy import asarray
  from numpy import save
  from numpy import load

  minibatch_size = 8
  counter = 0
  dataset_name = "mnist"

  listX = []
  listY = []
  for index in range(X_train.shape[0]):#range(10):#range(X_train.shape[0]):
    print("index = " + str(index))
    #if (counter >= 0 and counter < 1):
    listX.append(X_train[index])
    listY.append(Y_train[index]);
    if ((index + 1) % minibatch_size == 0) : 
      xKey = "mnist_8_" + str(counter) + "_x.npy"
      print("xKey " + xKey)
      #save_data_http(np.array(listX).tobytes(), url_data_server, xKey)
      save_data_http(listX, url_data_server, xKey)
      yKey = "mnist_8_" + str(counter) + "_y.npy"
      print("yKey " + yKey)
      #save_data_http(np.array(listY).tobytes(), url_data_server, yKey)      
      save_data_http(listY, url_data_server, yKey)  
      counter = counter + 1
      listX = []
      listY = []


is_remote = False
try:
  is_remote = (sys.argv[1].lower() == 'true')
  print("REMOTE HOST")
except:
  print("LOCAL HOST")

url_data_server = US.get_url_data(J.DEFAULT_JOB, is_remote)
print("url_data_server = " + url_data_server)
init_dataset_mnist_http(url_data_server)
