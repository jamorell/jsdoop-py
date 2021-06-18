import sys
import tensorflow as tf
import json
import time
import logging

#logging.basicConfig(format='%(asctime)s - %(processName)s  - %(filename)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
#logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)
logging.basicConfig(
    #filename='HISTORYlistener.log',
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
from src.api.load_job import load_job
from src.api.load_topology import load_model_topology_http
from src.api.load_current_weights import load_current_weights_http
from src.api.save_weights import save_model_weights_http
from src.api.load_gradients import load_gradients_http
from src.api.delete_gradients import delete_gradients_http
from src.helper import url_solver as US
from src.constants import constants as C

id_job = None
try:
  id_job = int(sys.argv[1]) #1234567890
except:
  exit("ERROR: Please insert a valid numeric job id.")


is_remote = False
try:
  is_remote = (sys.argv[2].lower() == 'true')
  logging.info("REMOTE HOST")
except:
  logging.info("LOCAL HOST")


username = "aggregator"

# LOAD JOB
url_job = US.get_url_job(is_remote)
job_json = load_job(url_job, id_job)



# LOAD MODEL TOPOLOGY
url_topology = US.get_url_topology(job_json, is_remote)
model_topology_key = job_json["topology"]["topology"]
mymodel = load_model_topology_http(url_topology, model_topology_key, id_job, username)
mymodel.summary()


### LOAD MODEL CURRENT WEIGHTS
age_model = -1
url_current_weights = US.get_url_current_weights(job_json, is_remote)
age_model = load_current_weights_http(mymodel, age_model, url_current_weights, id_job, username, 0)
age_model = int(age_model)



url_gradients_server = US.get_url_gradients(job_json, is_remote)

resultsQueue = "grads_" + str(id_job)

url_delete_gradients = US.get_url_delete_gradients(job_json, is_remote)

### OPTIMIZER
#optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01)
#optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
optimizer = tf.keras.optimizers.deserialize(job_json["optimizer"])
print(tf.keras.optimizers.serialize(optimizer))

finalGrads = None
counter = 0
adaptativeAggregation = job_json["aggregator"]["adaptative_aggregation"] # True # job_json["aggregator"]["adaptative_aggregation"]
gradientsToAccumulate = job_json["aggregator"]["gradients_to_accumulate"] #32 #8
minGradsToAccumulate = job_json["aggregator"]["min_grads_to_accumulate"] #2 # job_json["aggregator"]["min_grads_to_accumulate"]
maxGradsToAccumulate = job_json["aggregator"]["max_grads_to_accumulate"] #64 # job_json["aggregator"]["min_grads_to_accumulate"]
maxOutdatedGradientsBeforeACK = 32
gradientKeysToRemove = []
toAck = []

outdatedGradientKeysToRemove = []
outdatedToAck = []

limit_outdated_gradients = job_json["aggregator"]["limit_outdated_gradients"] #5 #2

def callback(ch, method, properties, body):
    global finalGrads
    global counter
    global gradientsToAccumulate
    global age_model
    global jobId
    global gradientKeysToRemove
    global toAck
    global url_gradients_server

    logging.debug("method " + str(method))
    logging.debug(" [x] Received %r" % body)

    try:
      ### Trying to load JSON
      myjson = json.loads(body)

      id_task = int(round(time.time() * 1000))

      logging.debug("key = " + str(myjson["key"]))
      age_gradients = myjson["key"][0 : myjson["key"].index("_")] 
      logging.debug("age_model = " + str(age_model))
      logging.debug("age_gradients = " + str(age_gradients))
      ### Checking old gradients
      if (int(age_gradients) >= (age_model - limit_outdated_gradients)):
        ###### ADAPTATIVE AGGREGATION INIT
        if adaptativeAggregation:
          gradientsToAccumulate = myjson["nWorkers"] # + 1;
          logging.info("BEFORE*#- nWorkers gradientsToAccumulate = " + str(gradientsToAccumulate)) 
          if (gradientsToAccumulate > maxGradsToAccumulate):
            gradientsToAccumulate = maxGradsToAccumulate;
          elif (gradientsToAccumulate < minGradsToAccumulate):
            gradientsToAccumulate = minGradsToAccumulate;    
          logging.info("AFTER*#- nWorkers gradientsToAccumulate = " + str(gradientsToAccumulate))     
        ###### ADAPTATIVE AGGREGATION END

        toAck.append(method.delivery_tag)
        gradientKeysToRemove.append(myjson["key"])
        #modelKeysToRemove.append(age_model)
        ### Loading gradients content --> If error then ACK
        try:
          logging.debug("Getting " + url_gradients_server + "?key=" + myjson["key"])
          gradients = load_gradients_http(age_model, url_gradients_server, id_job, myjson["key"], username, id_task)
          if (gradients is not None):
            logging.debug("type(gradients) = " + str(type(gradients)))

            for key, val in gradients.items():
              logging.debug("gradients key = " + key)


            if (finalGrads is None):
              finalGrads = []
              for k in range(len(gradients["gradients"])):
                mytensor = tf.convert_to_tensor(gradients["gradients"][k])
                logging.debug("str(mytensor.shape) " + str(k) + " -> " + str(mytensor.shape))

                #finalGrads.append(tf.convert_to_tensor(gradients["gradients"][k], dtype=tf.float32))
                finalGrads.append(mytensor)
                if (C.DEBUG): ##DEBUG
                  logging.debug(")))))))" + str(mytensor))
                  
            else:
               for k in range(len(gradients["gradients"])):
                mytensor = tf.convert_to_tensor(gradients["gradients"][k])
                logging.debug("str(mytensor.shape) " + str(k) + " -> " + str(mytensor.shape))
                #finalGrads[k] = tf.math.add(finalGrads[k], tf.convert_to_tensor(gradients["gradients"][k], dtype=tf.float32))
                finalGrads[k] = tf.math.add(finalGrads[k], mytensor)
                if (C.DEBUG): ##DEBUG
                  logging.debug("****" + str(finalGrads[k]))


            logging.debug(type(finalGrads))
            counter = counter + 1
            logging.debug("counter " + str(counter))
            if (counter >= gradientsToAccumulate):
              logging.debug("\n\n\n###ACCUMULATING*#-  " + str(counter) + " GRADIENTS")
              logging.info("AGG*#- nWorkers gradientsToAccumulate = " + str(gradientsToAccumulate))  
              start_time = int(round(time.time() * 1000))
              #counter = 0

              # Divide
              for i in range (len(finalGrads)): 
                finalGrads[i] = tf.divide(finalGrads[i], counter)          
                #finalGrads[i] = tf.divide(finalGrads[i], gradientsToAccumulate)

              finalGradsDictionary = {}
              for i in range(len(gradients["layers"])):
                finalGradsDictionary[gradients["layers"][i]] = gradients["gradients"][i]

              finalGradsList = []


              for i in range(len(mymodel.trainable_variables)):
                newName = mymodel.trainable_variables[i].name[0:mymodel.trainable_variables[i].name.index(":")]
                finalGradsList.append(finalGradsDictionary[newName])

                      
              optimizer.apply_gradients(zip(finalGradsList, mymodel.trainable_variables), experimental_aggregate_gradients=True) 

              logging.debug("applied gradients")

              # Save Model
              age_model = age_model + 1

              logging.debug("\n\n\nSAVING MODEL WEIGHTS " + str(age_model))

              names = [weight.name for layer in mymodel.layers for weight in layer.weights]
              weights = mymodel.trainable_variables
              names = list(map(lambda x: x[0:x.index(':')], names))
              execution_time = int(round(time.time() * 1000)) - start_time
              save_model_weights_http(weights, names, url_current_weights, id_job, age_model, False, gradientKeysToRemove, username, id_task, execution_time, counter) # counter = n_accumulated_gradients
              logging.debug("age_model = " + str(age_model))
              finalGrads = None
              
              for i in range(len(toAck)):
                logging.debug("ACK " + str(toAck[i]))
                channel.basic_ack(toAck[i])

              toAck.clear()
              gradientKeysToRemove.clear()
              counter = 0  
          else:
            print("age_model = " + str(age_model))
            print("ACK " + str(method.delivery_tag))
            #channel.basic_ack(method.delivery_tag)
            outdatedToAck.append(method.delivery_tag)
            outdatedGradientKeysToRemove.append(myjson["key"]) # these are not outdated, these are error gradients but I use the same array

        except Exception as e: 
          logging.debug("Exception getting data from json: " + str(e))      
      else:
        ### ACK old gradients
        logging.debug("age_model = " + str(age_model))
        logging.debug("too old gradients " + str(age_gradients))
        outdatedToAck.append(method.delivery_tag)
        outdatedGradientKeysToRemove.append(myjson["key"])
        if (len(outdatedGradientKeysToRemove) >= maxOutdatedGradientsBeforeACK):
          delete_gradients_http(id_job, outdatedGradientKeysToRemove, url_delete_gradients)
          for i in range(len(outdatedToAck)):
            logging.debug("ACK " + str(outdatedToAck[i]))
            channel.basic_ack(outdatedToAck[i])
          outdatedToAck.clear()
          outdatedGradientKeysToRemove.clear() 

    
    except Exception as e: 
      logging.debug("Exception loading json: " + str(e)) 






import pika

connection = pika.BlockingConnection(pika.ConnectionParameters( *US.get_rabbit_params(job_json) ))

channel = connection.channel()
channel.basic_qos(prefetch_count=(128)) # 1
#channel.basic_qos(prefetch_count=(gradientsToAccumulate * 2)) # 1


#3- Subscribe to Results Queue
channel.queue_declare(queue=resultsQueue, durable=True)
channel.basic_consume(queue=resultsQueue,
                      auto_ack=False,
                      consumer_tag="a-consumer-tag",  
                      on_message_callback=callback)

logging.info(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()

import signal
import sys

def signal_handler(sig, frame):
    logging.info('You pressed Ctrl+C!')
    channel.close()
    connection.close()    
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)
