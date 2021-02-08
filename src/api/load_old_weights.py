import requests
import io
import h5py
import tensorflow as tf
import numpy as np

def load_old_weights_http(model, age_model, url_current_weights, id_job):
  print("MAYBE WEIGHTS ARE DELETED SO WE HAVE TO TRY CATCH")
  print("################ load_old_weights_http INIT")
  try:
    PARAMS = {'id_job': id_job, 'age_model': age_model } 
    print("url_current_weights = " + url_current_weights)
    response = requests.get(url = url_current_weights, params = PARAMS) 
    
    print("len(response.content) = " + str(len(response.content)))

    print("response.headers = " + str(response.headers))


    if response.status_code == 200:
      print(response.content[0:200])
      print(response.status_code)
      responseGet = response

      from requests_toolbelt.multipart import decoder
      lst = []
      mydict = {}

      for part in decoder.MultipartDecoder(responseGet.content, responseGet.headers["Content-Type"]).parts:
          disposition = part.headers[b'Content-Disposition']
          params = {}
          for dispPart in str(disposition).split(';'):
              kv = dispPart.split('=', 2)
              params[str(kv[0]).strip()] = str(kv[1]).strip('\"\'\t \r\n') if len(kv)>1 else str(kv[0]).strip()
          type2 = part.headers[b'Content-Type'] if b'Content-Type' in part.headers else None
          #print("type = " + str(type2))
          #print("params = " + str(params))
         ## lst.append({'content': part.content, "type": type2, "params": params})

          #for key, value in params.items():
          #  print(key, value)
          #keys = params.items()
          keys = params.keys()

          if ("name" in keys):
            if (not str(params["name"]) in mydict):
              mydict[str(params["name"])] = []
            #else:
            #print("str(params[name]) =" + str(params["name"]))

            if (str(params["name"]) == "layers"):
              #print("is layers")

              layername = part.content.decode("utf-8")
              #print("BEFORE layername " + layername) 
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
              #print(mynumpy.shape)
              mydict[str(params["name"])].append(mynumpy)
              #print("is weights")
          
              #mydict[str(params["name"])].append(np.load(io.BytesIO(part.content), allow_pickle = True))
      #print(len(lst))
      #print(mydict.keys())
      #print(mydict["layers"])
      print("dictionary keys = " + str(mydict.keys()))
      mylayers = mydict["layers"]
      myweights = mydict["weights"]
      
      alllayers = model.layers
      #for i in range(len(alllayers)):
      #  print("alllayers[i] = " + alllayers[i].name)
      
      finaldict = {}
      for i in range(len(mylayers)):
        finaldict[mylayers[i]] = myweights[i]

      #model.set_weights(finaldict)
      #print("finaldict" + str(finaldict.keys()))
      for i in range(len(alllayers)):
        #print("length " + str(len( finaldict[mylayers[i]] )))
        #print("length " + str(len( model.get_layer(mylayers[i]) )))
        print(model.trainable_weights[i].name)
        finalkey = model.trainable_weights[i].name
        

        finalkey = finalkey[0:finalkey.index(':')]
        #print("WWWWWWWWWfinalkey " + finalkey)

        if finalkey in finaldict:
          #print("model shape = " + str(model.trainable_weights[i].shape))
          #print("remote model shape = " + str(finaldict[finalkey].shape))
          model.trainable_weights[i].assign(finaldict[finalkey])
          print(finalkey + " assigned") 
        else:
          print(model.trainable_weights[i].name + " NOT assigned")        

        #model.get_layer(mylayers[i]).weights.assign(finaldict[mylayers[i]])
  #    return response.headers["current_age"]
      #return ""
      return [response.headers["timeRequest"], response.headers["timeResponse"]]
    elif response.status_code == 204:
      return age_model
    else:
      raise Exception("Error: load_old_weights_http ") 
  except Exception as e:
    print("ERROR: Weights are already deleted. " + str(e))
    return None


