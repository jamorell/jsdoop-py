import requests
import io
import sys
from tempfile import SpooledTemporaryFile
import numpy as np
import h5py


from src.api.save_weights import save_model_weights_http
from src.api.load_job import load_job
from src.api.delete_job import delete_job
from src.helper import url_solver as US
from src.nn.topology import create_topology_mnist_28_28_1

id_job = None
try:
  id_job = int(sys.argv[1]) #1234567890
  print(id_job)
except:
  exit("ERROR: Please insert a valid numeric job id.")

username = "initiator"

# LOAD JOB
url_job = US.get_url_job()
job_json = load_job(url_job, id_job)

model = create_topology_mnist_28_28_1()

print(model.layers)
print(type(model.layers))
print(model.layers[0].name)

names = [weight.name for layer in model.layers for weight in layer.weights]
weights = model.get_weights()

names = list(map(lambda x: x[0:x.index(':')], names))

url_delete_job = US.get_url_delete_job()
delete_job(url_delete_job, id_job)
#requests.get(url = job_host + ":" + str(job_port) + "/delete_job", params = {"id_job": id_job})

url_current_weights = US.get_url_current_weights(job_json)
age_model = 1
#save_model_weights_http(weights, names, url_current_weights, id_job, age_model, True, None, None, username)
save_model_weights_http(weights, names, url_current_weights, id_job, age_model, True, None, username, 0, 0)

