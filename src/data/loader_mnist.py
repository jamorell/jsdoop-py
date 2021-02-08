from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.utils import to_categorical



def loader_mnist(myjson):
  (x_train, y_train), (x_test, y_test) = load_data()
  print(x_train.shape)
  print(y_train.shape)

  norm_x_train = x_train.astype(myjson["data"]["dtype"]) / 255 #x_train.astype('float32') / 255
  norm_x_test = x_test.astype(myjson["data"]["dtype"]) / 255


  encoded_y_train = to_categorical(y_train, num_classes=myjson["data"]["num_classes"], dtype=myjson["data"]["dtype"])
  encoded_y_test = to_categorical(y_test, num_classes=myjson["data"]["num_classes"], dtype=myjson["data"]["dtype"])

  X_train = norm_x_train.reshape(myjson["data"]["shape"]) #norm_x_train.reshape(-1, 28, 28, 1)
  Y_train = encoded_y_train
  X_test = norm_x_test.reshape(myjson["data"]["shape"]) #norm_x_test.reshape(-1, 28, 28, 1)
  Y_test = encoded_y_test
  return X_train, Y_train, X_test, Y_test
