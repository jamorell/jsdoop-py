import requests
import io
import h5py
import tensorflow as tf

def load_model_topology_http(url_model_topology, model_topology_key, id_job, username):
  PARAMS = {'key': model_topology_key, 'type': "h5", 'info_worker': "python", 'id_job': id_job, 'username': username }  
  response = requests.get(url = url_model_topology, stream = True,  params = PARAMS) 
  print(response)
  #print(response.raw.read())
  with h5py.File(io.BytesIO(response.raw.read()), 'r') as h5_file:
    model = tf.keras.models.load_model(h5_file)
    return model
