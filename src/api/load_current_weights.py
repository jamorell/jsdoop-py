import sys
import requests
import io
import h5py
import tensorflow as tf
import numpy as np
import logging
from src.constants import constants as C

from requests_toolbelt.multipart import decoder

logging.basicConfig(
    #filename='HISTORYlistener.log',
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

def load_current_weights_http(model, age_model, url_current_weights, id_job, username, id_task):
  logging.debug("################ load_current_weights_http INIT")
  PARAMS = {'id_job': id_job, 'age_model': age_model, 'info_worker': "python", "username": username, "id_task": id_task  } 
  logging.debug("url_current_weights = " + url_current_weights)
  response = requests.get(url = url_current_weights, params = PARAMS) 
  
  logging.debug("len(response.content) = " + str(len(response.content)))

  logging.debug("response.headers = " + str(response.headers))

  if response.status_code == 200:
    print(response.content[0:200])
    print(response.status_code)
    responseGet = response

    
    lst = []
    mydict = {}

    for part in decoder.MultipartDecoder(responseGet.content, responseGet.headers["Content-Type"]).parts:
        disposition = part.headers[b'Content-Disposition']
        params = {}
        for dispPart in str(disposition).split(';'):
            kv = dispPart.split('=', 2)
            params[str(kv[0]).strip()] = str(kv[1]).strip('\"\'\t \r\n') if len(kv)>1 else str(kv[0]).strip()
        type2 = part.headers[b'Content-Type'] if b'Content-Type' in part.headers else None
        if (C.DEBUG): ##DEBUG
          logging.debug("type = " + str(type2))
          logging.debug("params = " + str(params))
       ## lst.append({'content': part.content, "type": type2, "params": params})

        #for key, value in params.items():
        #  print(key, value)
        #keys = params.items()
        keys = params.keys()

        if ("name" in keys):
          if (not str(params["name"]) in mydict):
            mydict[str(params["name"])] = []
          #else:
          if (C.DEBUG): ##DEBUG
            logging.debug("str(params[name]) =" + str(params["name"]))

          if (str(params["name"]) == "layers"):
            if (C.DEBUG): ##DEBUG
              logging.debug("is layers")

            layername = part.content.decode("utf-8")
            if (C.DEBUG): ##DEBUG
              logging.debug("BEFORE layername " + layername) 
            #layername = layername + ":0"
#              index = layername.index(':')
            '''
            import re
            pattern = re.compile("\/.*:[0-9]*")
            index = pattern.search(layername).span()
            index = index[0]
            print("...................index = " + str(index))
            if (index > 0):
              layername = layername[0:index]
            '''
            mydict[str(params["name"])].append(layername)
          else:
            #mynumpy = np.load(io.BytesIO(part.content), allow_pickle = True)
            mynumpy = np.load(io.BytesIO(part.content), allow_pickle = False)
            if (C.DEBUG): ##DEBUG
              logging.debug(mynumpy.shape)
            mydict[str(params["name"])].append(mynumpy)
            if (C.DEBUG): ##DEBUG
              logging.debug("is weights")
        
            #mydict[str(params["name"])].append(np.load(io.BytesIO(part.content), allow_pickle = True))
    #print(len(lst))
    if (C.DEBUG): ##DEBUG
      logging.debug(mydict.keys())
      logging.debug(mydict["layers"])
    mylayers = mydict["layers"]
    myweights = mydict["weights"]
    
    alllayers = model.layers
    if (C.DEBUG): ##DEBUG
      for i in range(len(alllayers)):
        logging.debug("alllayers[i] = " + alllayers[i].name)
    
    finaldict = {}
    for i in range(len(mylayers)):
      finaldict[mylayers[i]] = myweights[i]

    #model.set_weights(finaldict)
    if (C.DEBUG): ##DEBUG
      logging.debug("finaldict" + str(finaldict.keys()))
    for i in range(len(alllayers)):
      #print("length " + str(len( finaldict[mylayers[i]] )))
      #print("length " + str(len( model.get_layer(mylayers[i]) )))
      if (C.DEBUG): ##DEBUG
        logging.debug(model.trainable_weights[i].name)
      finalkey = model.trainable_weights[i].name
      

      finalkey = finalkey[0:finalkey.index(':')]
      if (C.DEBUG): ##DEBUG
        logging.debug("WWWWWWWWWfinalkey " + finalkey)

      if finalkey in finaldict:
        if (C.DEBUG): ##DEBUG
          logging.debug("model shape = " + str(model.trainable_weights[i].shape))
          logging.debug("remote model shape = " + str(finaldict[finalkey].shape))
        model.trainable_weights[i].assign(finaldict[finalkey])
        if (C.DEBUG): ##DEBUG
          logging.debug(finalkey + " assigned") 
      else:
        if (C.DEBUG): ##DEBUG
          logging.debug(model.trainable_weights[i].name + " NOT assigned")        

      #model.get_layer(mylayers[i]).weights.assign(finaldict[mylayers[i]])
    logging.debug("################ load_current_weights_http END -> OK")
    return response.headers["current_age"]
    '''
    for i in range(len(mylayers)):
      print("mylayers[i] " + mylayers[i]) 
      print(model.get_layer(mylayers[i]))
      model.trainable_weights[i].assign(nparray[i])
      #model.get_layer(mylayers[i]).weights.assign(myweights[i])      
      #model.get_layer(mylayers[i]).assign(myweights[i])
      #model.trainable_weights[i].assign(nparray[i])
    '''
    
    #return response.headers["age_model"]

    '''
    nparray = np.load(io.BytesIO(response.content), allow_pickle = True)
    print(str(type(nparray)))
    for i in range(len(model.trainable_weights)):
      model.trainable_weights[i].assign(nparray[i])

    print(response.headers)
    print("################ load_current_weights_http END")
    return response.headers["age_model"]
    '''
    
  elif response.status_code == 204:
    logging.debug("################ load_current_weights_http END -> Nothing to update")
    return age_model
  else:
    logging.error("Error: load_current_weights_http trying to load current weights.  response.status_code = " + str(response.status_code)) 
    logging.debug("################ load_current_weights_http END -> Error updating ")
    return age_model

