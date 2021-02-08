import requests
import numpy as np
import tensorflow as tf
import json
import io
from tempfile import SpooledTemporaryFile
import h5py
import sys

from src.api.load_job import load_job
from src.api.load_topology import load_model_topology_http
from src.api.load_old_weights import load_old_weights_http
from src.helper import url_solver as US
from src.data.loader import loader

id_job = None
try:
  id_job = int(sys.argv[1]) #1606147964029
except:
  exit("ERROR: Please insert a valid numeric job id.")

username = "tester"


is_remote = False
try:
  is_remote = (sys.argv[2].lower() == 'true')
  print("REMOTE HOST")
except:
  print("LOCAL HOST")


# LOAD JOB
url_job = US.get_url_job(is_remote)
job_json = load_job(url_job, id_job)


# LOAD MODEL TOPOLOGY
url_topology = US.get_url_topology(job_json, is_remote)
model_topology_key = job_json["topology"]["topology"]
mymodel = load_model_topology_http(url_topology, model_topology_key, id_job, username)
mymodel.summary()



### LOADING DATA
X_train, Y_train, X_test, Y_test = loader(job_json)
###################


### LOAD MODEL OLD WEIGHTS
'''
if (debug):
  job_json["weights"]["host"] = "http://localhost"
  job_json["weights"]["port"] = 8081
url_old_weights = job_json["weights"]["host"] + ':' + str(job_json["weights"]["port"]) + '/old_weights'
'''
url_old_weights = US.get_url_old_weights(job_json, is_remote)

resultsQueue = "weights_" + str(id_job)

toAck = []

#m = tf.keras.metrics.CategoricalAccuracy()
#mse = tf.keras.losses.MeanSquaredError()
m = tf.keras.metrics.deserialize(job_json["tester"]["metric"])
mse = tf.keras.losses.deserialize(job_json["tester"]["losses"])

print(tf.keras.optimizers.serialize(m))
print(tf.keras.optimizers.serialize(mse))

def callback(ch, method, properties, body):
    global mymodel
    global toAck
    global url_old_weights
    global channel

    print("method " + str(method))
    print(" [x] Received %r" % body)
    myjson = json.loads(body)

    toAck.append(method.delivery_tag)

    ############
    print("Getting " + url_old_weights + "?id_job=" + str(myjson["idJob"]) + "?age_model=" + str(myjson["ageModel"]));
    try:
      times = load_old_weights_http(mymodel, myjson["ageModel"], url_old_weights, myjson["idJob"])
      print("BEFORE PREDICTING")
      y_pred = mymodel.predict(X_test) 
      
      print(y_pred)
      loss = mse(Y_test, y_pred).numpy()
      m.reset_states()
      
      m.update_state(Y_test, y_pred)
      acc = m.result().numpy()
      print("#### AGE MODEL " + str(myjson["ageModel"]))
      print("loss = " + str(loss))
      print("acc = " + str(acc))
      ############
      for i in range(len(toAck)):
        print("ACK " + str(toAck[i]))
        channel.basic_ack(toAck[i])

      toAck.clear()

      message =  '{ "idJob": ' + str(myjson["idJob"]) + ' , "ageModel":' + str(myjson["ageModel"]) + ', "loss":' + str(loss) + ', "acc":' + str(acc) + ', "requestTime":' + str(times[0]) + ', "responseTime":' + str(times[1]) + '}'
      queue_name = "test_result_" + str(myjson["idJob"])
      print("sending test results message = " + message)
      print("sending test results queue_name = " + queue_name)
      channel.queue_declare(queue = queue_name)
      channel.basic_publish(exchange = '', routing_key = queue_name, body = message)
    except:
      print("next!")
    



import pika

#connection = pika.BlockingConnection(pika.ConnectionParameters(host = job_json["aggregator"]["rabbit_host"], port = job_json["aggregator"]["rabbit_port"] ))
connection = pika.BlockingConnection(pika.ConnectionParameters( *US.get_rabbit_params(job_json) ))


channel = connection.channel()
channel.basic_qos(prefetch_count=(8)) # 1


#3- Subscribe to Results Queue
channel.queue_declare(queue=resultsQueue, durable=True)
channel.basic_consume(queue=resultsQueue,
                      auto_ack=False,
                      consumer_tag="a-consumer-tag",  
                      on_message_callback=callback)

print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()

import signal

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    channel.close()
    connection.close()    
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)
