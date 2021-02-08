import requests
import time
import io
import h5py
import tensorflow as tf
from tempfile import SpooledTemporaryFile
import numpy as np

def save_gradients_http(gradients, names, url_gradients, id_job, age_model, username, id_task, start_time ):
  #start_time = int(round(time.time() * 1000))
  listFiles = []
  print(type(gradients))
  
  for i in range(len(gradients)):
    outfile = SpooledTemporaryFile()
    np.save(outfile, tf.keras.backend.eval(gradients[i]))
    _ = outfile.seek(0)
    listFiles.append(('gradients', outfile))
    listFiles.append(('layers', names[i]))
  
  #print(type(tf.keras.backend.eval(weights[0])))
  #execution_time = execution_time + (int(round(time.time() * 1000)) - start_time)
  execution_time = (int(round(time.time() * 1000)) - start_time)
  PARAMS = {'id_job': id_job, 'age_model': age_model, 'info_worker': "python", "username": username, "id_task": id_task, "execution_time": execution_time}; 
  #headers = {'content-type': ''}
  r = requests.post(url_gradients,
  files = listFiles, params = PARAMS) 
  print(r)
  print(r.content[0:200])
  print(r.headers)
  return int(r.headers["current_age"])
