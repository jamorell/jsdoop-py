import requests
import numpy as np

def load_data_http(url_data_server, key, id_job, username, id_task):#, info_worker):
  PARAMS = {'key': key, "id_job": id_job, "username": username, "id_task": id_task, "info_worker": "python" } #, "info_worker": info_worker } 

  headers = {'content-type': 'application/octet-stream'}
  response = requests.get(url = url_data_server, stream = True,  params = PARAMS) 
  thebytes = response.raw.read()
  thearray = np.frombuffer(thebytes)
  return thearray

'''
def load_data_http(url_data_server, key, id_job, username):
  key = "mnist_8_6339_y.npy"
  print("key = " + key)
  print("url_data_server = " + url_data_server)
  PARAMS = {'key': key, "id_job": id_job, "username": username } 

  headers = {'content-type': 'application/octet-stream'}
  response = requests.get(url = url_data_server, stream = True,  params = PARAMS, headers=headers) 
  #print(response)
  #print(response.raw.read())
  #print("response.raw.read() " + str(response.raw.read()))
  print("response = " + str(response.headers))
  mybuffer = io.BytesIO(response.raw.read())
  print(str(type(mybuffer)))
  print(str(mybuffer.getbuffer().nbytes))
  print(str(mybuffer.getbuffer()))
  returnNumpy = np.load(mybuffer, allow_pickle = True)
  exit()
  return ""
#  mybuffer = response.raw.read()
#  inputfile = SpooledTemporaryFile()
#  inputfile.write(mybuffer)
#  _ = inputfile.seek(0)
#  returnNumpy = np.load(inputfile)
#  #returnNumpy = np.load(io.BytesIO(), allow_pickle = True)
#  print(returnNumpy)
#  return returnNumpy
'''
