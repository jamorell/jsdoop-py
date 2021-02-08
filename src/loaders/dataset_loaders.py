import tensorflow as tf
import random 

from src.helper import url_solver as US
from src.data.loader import loader
from src.data.create_batches import create_batches, get_total_m_batches


class DatasetLoaderJSON:
  def __init__(self, json, seed):
    if seed is not None:
      random.seed(seed)
      print("seed = " + str(seed))

    self.loss_object = tf.keras.losses.deserialize(json["tester"]["losses"])
    self.X_train, self.y_train, self.X_test, self.y_test = loader(json)
    self.X_batches, self.y_batches = create_batches(self.X_train, self.y_train, json)

    ### LOCAL DATASET
    self.total_mbatches = get_total_m_batches(json)
    self.local_dataset_len = int(self.total_mbatches / json["data"]["local_portion_dataset"])
    self.local_dataset = random.sample(range(self.total_mbatches), self.local_dataset_len)
    print("self.local_dataset = "+ str(self.local_dataset))


  def get_random_batch_from_local_dataset(self):
    mb_selected = random.randint(0, len(self.local_dataset) - 1) #str(i)
    xs = self.X_batches[self.local_dataset[mb_selected]] #load_data_http(url_data_server, "mnist_8_" + str(mb_selected) + "_x.npy")
    ys = self.y_batches[self.local_dataset[mb_selected]] #load_data_http(url_data_server, "mnist_8_" + str(mb_selected) + "_y.npy")
    return xs, ys, mb_selected
