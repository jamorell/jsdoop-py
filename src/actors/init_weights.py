import sys

from src.api.load_job import load_job
from src.api.load_topology import load_model_topology_http
from src.api.load_current_weights import load_current_weights_http
from src.api.save_weights import save_model_weights_http

id_job = None
try:
  id_job = int(sys.argv[1]) #1606147964029
except:
  exit("ERROR: Please insert a valid numeric job id.")

username = None
try:
  username = "worker_py_" + str(sys.argv[2])
except:
  exit("ERROR: Please insert a valid username.")

debug = False
try:
  debug = (sys.argv[3].lower() == 'true')
  job_host = "http://localhost"
  job_port = 8081
  print("LOCAL HOST")
except:
  job_host = "http://mallba9.lcc.uma.es"
  job_port = 4042
  print("MALLBA9 HOST")



# LOAD JOB
url_job = job_host + ":" + str(job_port) + "/get_job"
myjson = load_job(url_job, id_job)
print(myjson["id_job"])
print(myjson["topology"])

print(myjson["topology"]["host"])


# LOAD MODEL TOPOLOGY
#testing
if (debug):
  myjson["topology"]["host"] = "http://localhost"
  myjson["topology"]["port"] = 8081
#testing
model_topology_key = myjson["topology"]["topology"] #"another_model" #"mnist_28_28_1.h5"
url_topology = myjson["topology"]["host"] + ":" + str(myjson["topology"]["port"]) + "/topology"
print(url_topology)
model = load_model_topology_http(url_topology, model_topology_key, id_job, username)
model.summary()


# SAVING INIT WEIGHTS #Saving Initial Weights. This method saves the initial weights if they do not exist, otherwise, they must be removed first.
#testing
if (debug):
  myjson["weights"]["host"] = "http://localhost"
  myjson["weights"]["port"] = 8081

age_model = 1
url_current_weights = myjson["weights"]["host"] + ":" + str(myjson["weights"]["port"]) + "/current_weights"
names = [weight.name for layer in model.layers for weight in layer.weights]
weights = model.get_weights()
names = list(map(lambda x: x[0:x.index(':')], names))

#for name, weight in zip(names, weights):
#    print(name, weight.shape, "hola")

save_model_weights_http(weights, names, url_current_weights, id_job, age_model, True, None, username, 0, 0)

