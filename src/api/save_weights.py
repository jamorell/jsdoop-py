import requests
import time
import io
import h5py
import tensorflow as tf
from tempfile import SpooledTemporaryFile
import numpy as np


def save_model_weights_http(weights, names, url_current_weights, id_job, age_model, force_save, todelete_ids_gradients, username, id_task, execution_time, n_accumulated_gradients = 0):
 
  listFiles = []
  print(type(weights))
  
  for i in range(len(weights)):
    outfile = SpooledTemporaryFile()
    np.save(outfile, tf.keras.backend.eval(weights[i]))
    _ = outfile.seek(0)
    listFiles.append(('weights', outfile))
    listFiles.append(('layers', names[i]))
  if todelete_ids_gradients:
    for i in range(len(todelete_ids_gradients)):
      listFiles.append(('todelete_grads', todelete_ids_gradients[i]))


  #print(type(tf.keras.backend.eval(weights[0])))


  PARAMS = {'id_job': id_job, 'age_model': age_model, 'info_worker': "python_aggregator", 'username': username, "id_task": id_task, "execution_time": execution_time, "n_accumulated_gradients": n_accumulated_gradients}#, 'id_grads_to_delete': todelete_ids_gradients}; 
  #r = requests.post("http://localhost:8082/jamorell/JSDoop/1.1.0/testing_weights",
  #r = requests.post("http://localhost:8081/current_weights",
  r = requests.post(url_current_weights,
  files = listFiles, params = PARAMS) 
  print(r)
  print(r.content[0:200])
  print(r.headers)




'''
def save_model_weights_http(weights, names, url_current_weights, id_job, age_model, force_save, todelete_ids_gradients, todelete_ids_weights, username):
 
  listFiles = []
  print(type(weights))
  
  for i in range(len(weights)):
    outfile = SpooledTemporaryFile()
    np.save(outfile, tf.keras.backend.eval(weights[i]))
    _ = outfile.seek(0)
    #print(outfile.read())
    #print(names[i])
    #print(np.frombuffer(outfile.read(), np.uint8))
    #import binascii
    #print(binascii.hexlify(outfile.read()))
    #testingbuffer = outfile.read()
    #int_values = [x for x in testingbuffer]
    #print(list(testingbuffer))
    #print(list(bytearray(testingbuffer)))
    #print("adios")
    #quit()
    listFiles.append(('weights', outfile))
   # listFiles.append(('layers', names[i].encode('utf_8')))
    #listFiles.append(('layers', bytes([0x13, 0x00, 0x00, 0x00, 0x08, 0x00])))
    listFiles.append(('layers', names[i]))

  print(type(tf.keras.backend.eval(weights[0])))

#  listFiles.append(('layers', names))

  PARAMS = {'id_job': id_job, 'age_model': age_model, 'force_save': force_save, 'info_worker': "python", 'username': username, "id_task":int(round(time.time() * 1000)) }; 
  #r = requests.post("http://localhost:8082/jamorell/JSDoop/1.1.0/testing_weights",
  r = requests.post("http://localhost:8081/current_weights",
  files = listFiles, params = PARAMS) 
  print(r)
  print(r.content[0:200])
  print(r.headers)

  if (r.status_code == 201 or r.status_code == 409):
    current_age = r.headers["current_age"]
    print(current_age)

    PARAMS["age_model"] = int(current_age) - 1;

    responseGet = requests.get(url = "http://localhost:8081/current_weights", params = PARAMS) 

    print(str(responseGet.content)[0:500])
    print("HEADERS")
    print(str(responseGet.headers))
    print(responseGet)
    #print(str(responseGet.content))

    #https://stackoverflow.com/questions/50925083/parse-multipart-request-string-in-python
    #https://stackoverflow.com/questions/57790416/how-to-access-field-names-in-multipartdecoder
    from requests_toolbelt.multipart import decoder
    #decoder = decoder.MultipartDecoder(responseGet.content, responseGet.headers["Content-Type"])
    #print(decoder)
  

    lst = []
    mydict = {}
    for part in decoder.MultipartDecoder(responseGet.content, responseGet.headers["Content-Type"]).parts:
        disposition = part.headers[b'Content-Disposition']
        params = {}
        for dispPart in str(disposition).split(';'):
            kv = dispPart.split('=', 2)
            params[str(kv[0]).strip()] = str(kv[1]).strip('\"\'\t \r\n') if len(kv)>1 else str(kv[0]).strip()
        type2 = part.headers[b'Content-Type'] if b'Content-Type' in part.headers else None
        print("type = " + str(type2))
        print("params = " + str(params))
       ## lst.append({'content': part.content, "type": type2, "params": params})

        #for key, value in params.items():
        #  print(key, value)
        #keys = params.items()
        keys = params.keys()
        if ("name" in keys):
          if (not str(params["name"]) in mydict):
            mydict[str(params["name"])] = []
          else:
            mydict[str(params["name"])].append(part.content)
    #print(len(lst))
    print(mydict.keys())
    print(mydict["layers"])
'''
