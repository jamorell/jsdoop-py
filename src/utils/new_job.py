import requests
import io
import sys
from tempfile import SpooledTemporaryFile
import numpy as np
import h5py

from src.api.create_job import create_job
from src.api.save_weights import save_model_weights_http
from src.api.load_job import load_job
from src.helper import url_solver as US
from src.nn.topology_initiator import create_topology
from src.constants import jobs as J

username = "initiator"

is_remote = False
try:
  is_remote = (sys.argv[1].lower() == 'true')
except:
  print("LOCAL HOST")

if is_remote:
  print("REMOTE HOST")
else:
  print("LOCAL HOST")


#url_job = HOST + ":" + PORT
url_job = US.get_url_save_job(is_remote)
id_job = create_job(J.DEFAULT_JOB, url_job)

model = create_topology(J.DEFAULT_JOB["topology"]["topology"])#create_topology_mnist_28_28_1()

names = [weight.name for layer in model.layers for weight in layer.weights]
weights = model.get_weights()

names = list(map(lambda x: x[0:x.index(':')], names))

url_current_weights = US.get_url_current_weights(J.DEFAULT_JOB, is_remote)
print("URL = " + url_current_weights)

age_model = 1
save_model_weights_http(weights, names, url_current_weights, id_job, age_model, True, None, username, 0, 0)

print(id_job)
