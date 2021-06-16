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







def onehot_to_int(one_hot):
  if (np.isscalar(one_hot)):
    return one_hot
  for i in range(len(one_hot)):
    if (one_hot[i] == 1):
      return i







##### 3. Converting and dividing balanced CIFAR10 into a non-IID dataset. https://towardsdatascience.com/preserving-data-privacy-in-deep-learning-part-3-ae2103c40c22
def print_split(clients_split, n_labels): 
  total_data = 0
  print("Data split:")
  for i, client in enumerate(clients_split):
    temp = []
    print("len(client[1]) = " + str(len(client[1])))
    for k in len(client[1]):
      print("client[1][k] = " + str(client[1][k]))
      temp.append(onehot_to_int(client[1][k]))
    print("temp = " + str(temp))
    split = np.sum(temp.reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
    print(" - Client {}: {} Total = {}".format(i, split, np.sum(split)))
    total_data = total_data + np.sum(split)

  print("total_data = " + str(total_data))
  print()

def clients_rand(train_len, nclients):
  '''
  train_len: size of the train data
  nclients: number of clients
  
  Returns: to_ret
  
  This function creates a random distribution 
  for the clients, i.e. number of images each client 
  possess.
  '''
  ### LOCAL RANDOM #### 
  local_seed = 0
  state = np.random.get_state()
  np.random.seed(local_seed)
  #####################
  client_tmp=[]
  sum_=0
  #### creating random values for each client ####
  for i in range(nclients):
    tmp=np.random.randint(10,100)
    sum_+=tmp
    client_tmp.append(tmp)



  client_tmp= np.array(client_tmp)
  #### using those random values as weights ####
  clients_dist= ((client_tmp/sum_)*train_len).astype(int)
  num  = train_len - clients_dist.sum()

  ### Add the rest to each client ####
  while (num > 0):
    for i in range(nclients):
      if (num > 0):
        clients_dist[i] = clients_dist[i] + 1
        num = num - 1
      else:
        break

  to_ret = list(clients_dist)
  ### LOCAL RANDOM #### 
  np.random.set_state(state)
  #####################  
  return to_ret


def iid_to_noniid_data(data, labels, n_clients=100, classes_per_client=10, shuffle=True, verbose=True):
  '''
  Splits (data, labels) among 'n_clients s.t. every client can holds 'classes_per_client' number of classes
  Input:
    data : [n_data x shape]
    labels : [n_data (x 1)] from 0 to n_labels
    n_clients : number of clients
    classes_per_client : number of classes per client
    shuffle : True/False => True for shuffling the dataset, False otherwise
    verbose : True/False => True for printing some info, False otherwise
  Output:
    clients_split : client data into desired format
  '''
  ### LOCAL RANDOM #### 
  local_seed = 0
  state = np.random.get_state()
  np.random.seed(local_seed)
  #####################
  
  print("labels = " + str(labels))
  print("labels.shape = " + str(labels.shape))

  #### constants #### 
  n_data = data.shape[0]
  n_labels = 10  # TODO #int(np.max(labels) + 1)
  print("n_labels = " + str(n_labels))


  ### client distribution ####
  data_per_client = clients_rand(len(data), n_clients)
  data_per_client_per_class = [np.maximum(1,nd // classes_per_client) for nd in data_per_client]
  
  # sort for labels
  data_idcs = [[] for i in range(n_labels)]
  for j, label in enumerate(labels):
    print("j =" + str(j))
    print("label =" + str(label))
    print("n_labels = " + str(n_labels))
    print("one_hot_to_int = " + str(onehot_to_int(label)))
    data_idcs[onehot_to_int(label)] += [j]
    #data_idcs[label] += [j]
  
  print("data_idcs = " + str(data_idcs))
  '''
  if shuffle:
    for idcs in data_idcs:
      np.random.shuffle(idcs)
  '''
  for idcs in data_idcs:
    idcs.sort()
    print("idcs = " + str(idcs))
    
  # split data among clients
  clients_split = []
  c = 0
  for i in range(n_clients):
    client_idcs = []
        
    budget = data_per_client[i]
    c = np.random.randint(n_labels)
    while budget > 0:
      take = min(data_per_client_per_class[i], len(data_idcs[c]), budget)
      #print("take = " + str(take))
      
      client_idcs += data_idcs[c][:take]
      data_idcs[c] = data_idcs[c][take:]
      
      budget -= take
      c = (c + 1) % n_labels
      
    print("before adding to clients_split client_idcs = " + str(client_idcs))  
    clients_split += [(data[client_idcs], labels[client_idcs])]

  '''
  print("len(client_idcs) " + str(len(client_idcs)))
  for idcs in client_idcs:
    client_idcs.sort()
    print("client_idcs = " + str(idcs))
  '''
  clients_split = np.array(clients_split)
  
  ### LOCAL RANDOM #### 
  np.random.set_state(state)
  #####################

  return clients_split


class DatasetLoaderNonIID:
  def __init__(self, json, seed, is_remote):
    if seed is not None:
      random.seed(seed)
      print("seed = " + str(seed))

    self.loss_object = tf.keras.losses.deserialize(json["tester"]["losses"])
    self.X_train, self.y_train, self.X_test, self.y_test = loader(json)

    #seed in this case is the ID of the worker
    n_labels = 10 # TODO
    n_clients = json["data"]["local_portion_dataset"]
    dataset_split_noniid = iid_to_noniid_data(self.X_train, self.y_train, n_clients=64, classes_per_client=3, shuffle=True, verbose=True)
    print("dataset_split_noniid[" + str(seed) + "][1] = " + str(dataset_split_noniid[seed][1]))
    print(dataset_split_noniid.shape)
    print_split(dataset_split_noniid, n_labels)
    print("seed = " + str(seed))
    #self.X_batches, self.y_batches = create_batches(self.X_train, self.y_train, json)
    self.X_batches, self.y_batches = create_batches(dataset_split_noniid[seed][0], dataset_split_noniid[seed][1], json)
    print("self.y_batches = " + str(self.y_batches))
    ### LOCAL DATASET
    self.total_mbatches = get_total_m_batches(json)
    self.local_dataset_len = len(self.X_batches)  
    exit()


  def get_random_batch(self, worker, id_task, start_time):
    mb_selected = np.random.randint(0, len(self.X_batches)) 
    xs = self.X_batches[mb_selected]
    ys = self.y_batches[mb_selected]
    return xs, ys, mb_selected

