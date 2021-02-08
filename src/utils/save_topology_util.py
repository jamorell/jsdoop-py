import sys

from src.nn.topology_initiator import create_topology
from src.api.save_topology import save_model_topology_http
from src.constants import jobs as J
from src.helper import url_solver as US

topology_name = None
try:
  topology_name = sys.argv[1]
except:
  exit("ERROR: Please insert a valid topology name. See /src/nn/topology_initiator "  + sys.argv[1])

is_remote = False
try:
  is_remote = (sys.argv[2].lower() == 'true')
  print("REMOTE HOST")
except:
  print("LOCAL HOST")

username = "initiator"

model = create_topology(topology_name)

if (model == None):
  exit("2ERROR: Please insert a valid topology name. See /src/nn/topology_initiator "  + sys.argv[1])


url_model_topology = US.get_url_topology(J.DEFAULT_JOB, is_remote)
save_model_topology_http(model, url_model_topology, topology_name)
