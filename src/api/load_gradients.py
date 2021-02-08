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

def load_gradients_http(age_model, url_gradients, id_job, id_gradients, username, id_task):
  print("################ load_gradients INIT")
  PARAMS = {'id_job': id_job, 'age_model': age_model, 'id_grads': id_gradients, 'info_worker': "python_aggregator", "username": username, "id_task": id_task } 
  print("url_gradients = " + url_gradients)
  response = requests.get(url = url_gradients, params = PARAMS) 
  print(response)
  print(response.status_code)
  print("len(response.content) = " + str(len(response.content)))
  if response.status_code == 200:

    print(response.content[0:200])
    #'''
    responseGet = response

    
    lst = []
    mydict = {}
    print("TOTAL PARTS  = " + str(len(decoder.MultipartDecoder(responseGet.content, responseGet.headers["Content-Type"]).parts)))
    totalLayersTest = 0
    totalGradientsTest = 0
    try:
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
              totalLayersTest = totalLayersTest + 1
              if (C.DEBUG): ##DEBUG
                logging.debug("is layers")

              layername = part.content.decode("utf-8")
              if (C.DEBUG): ##DEBUG
                logging.debug("BEFORE layername " + layername) 
  #              index = layername.index(':')
              #
              #import re
              #pattern = re.compile("\/.*:[0-9]*")
              #index = pattern.search(layername).span()
              #index = index[0]
              #print("...................index = " + str(index))
              #if (index > 0):
              #  layername = layername[0:index]
              #
              mydict[str(params["name"])].append(layername)
            else:
              totalGradientsTest = totalGradientsTest + 1
              #mynumpy = np.load(io.BytesIO(part.content), allow_pickle = True)
              mynumpy = np.load(io.BytesIO(part.content), allow_pickle = False)
              if (C.DEBUG): ##DEBUG
                logging.debug(mynumpy.shape)
                logging.debug("saving in key " + str(params["name"]))

              mydict[str(params["name"])].append(mynumpy)

              if (C.DEBUG): ##DEBUG
                logging.debug("is weights")

      if (C.DEBUG): ##DEBUG
        logging.debug("mydict keys = " + str(mydict.keys() ))
        logging.debug("TOTAL PARTS len(mydict layers) = " + str(len(mydict["layers"])))
        logging.debug(".-.TOTAL PARTS len(mydict gradients) = " + str(len(mydict["gradients"])))
        logging.debug("TOTAL PARTS len(mydict totalLayersTest) = " + str(totalLayersTest))
        logging.debug("TOTAL PARTS len(mydict totalGradientsTest) = " + str(totalGradientsTest))

      return mydict
      #'''
      #return 0
    except Exception as e:
      logging.error("EXCEPTION loading gradients ->  exception = " + str(e))
      return None

