import tensorflow as tf
import random 
import numpy as np

from src.helper import url_solver as US
from src.data.loader import loader
from src.data.create_batches import create_batches, get_total_m_batches
from src.api.load_data import load_data_http


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


class DatasetLoaderHTTP:
  def __init__(self, json, seed, is_remote):
    if seed is not None:
      random.seed(seed)
      print("seed = " + str(seed))

    self.url = US.get_url_data(json, is_remote) #US.get_url_data(json, is_remote) 
    self.x_shape = json["data"]["shape"]
    self.num_classes = json["data"]["num_classes"]
    self.dtype = json["data"]["dtype"]
    self.mb_size = json["data"]["mb_size"]

    self.loss_object = tf.keras.losses.deserialize(json["tester"]["losses"])

    ### LOCAL DATASET
    self.total_mbatches = get_total_m_batches(json)
    self.local_dataset_len = int(self.total_mbatches / json["data"]["local_portion_dataset"])
    self.local_dataset = random.sample(range(self.total_mbatches), self.local_dataset_len)
    print("self.local_dataset = "+ str(self.local_dataset))

  def load_x_data(self, x_shape, dtype, url_data_server, key, id_job, username, id_task): #, info_worker):
    mydata = load_data_http(url_data_server, key, id_job, username, id_task) #, info_worker)
    mydata = np.frombuffer(mydata, dtype) 
    mydata = mydata.reshape(x_shape)
    return mydata
  
  def load_y_data(self, num_classes, dtype, url_data_server, key, id_job, username, id_task): #, info_worker):
    mydata = load_data_http(url_data_server, key, id_job, username, id_task) #, info_worker)
    mydata = np.frombuffer(mydata, dtype) 
    mydata = mydata.reshape(-1, num_classes)
    return mydata

  def get_random_batch(self, worker, id_task, start_time):
    mb_selected = random.randint(0, len(self.local_dataset) - 1) #str(i)
    x_key = "mnist_" + str(self.mb_size) + "_" + str(self.local_dataset[mb_selected]) + "_x.npy"
    y_key = "mnist_" + str(self.mb_size) + "_" + str(self.local_dataset[mb_selected]) + "_y.npy"
    xs = self.load_x_data(self.x_shape, self.dtype, self.url, x_key, worker.id_job, worker.username, id_task)#, worker.info_worker)
    ys = self.load_y_data(self.num_classes, self.dtype, self.url, y_key, worker.id_job, worker.username, id_task)#, worker.info_worker)
    return xs, ys, mb_selected
