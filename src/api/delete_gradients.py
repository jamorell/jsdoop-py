import requests
import io

def delete_gradients_http(id_job, todelete_ids_gradients, url_delete_gradients):
  listFiles = []
  for i in range(len(todelete_ids_gradients)):
    listFiles.append(('todelete_grads', todelete_ids_gradients[i]))
  PARAMS = {'id_job': id_job}
  #r = requests.post("http://localhost:8081/delete_gradients",
  r = requests.post(url_delete_gradients,
  files = listFiles, params = PARAMS) 
  print(r)
  print(r.content[0:200])
  print(r.headers)
