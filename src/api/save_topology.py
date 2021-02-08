import requests
import time
import io
import h5py
import tensorflow as tf
from tempfile import SpooledTemporaryFile
import numpy as np

def save_model_topology_http(model, url_model_topology, key):
  myfile = SpooledTemporaryFile()
  h5_file = h5py.File(myfile)
  tf.keras.models.save_model(model = model, filepath = h5_file)
  _ = myfile.seek(0)
  #r = requests.post("http://localhost:8081/topology?key=" + key,
  r = requests.post(url_model_topology + "?key=" + key,
                      data = myfile,
                      headers = { 'Content-Type': 'application/octet-stream' })
  print(r)
