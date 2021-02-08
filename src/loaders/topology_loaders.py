from src.helper import url_solver as US
from src.api.load_topology import load_model_topology_http


class TopologyLoaderHTTP:
  def __init__(self, json, is_remote):
    self.url = US.get_url_topology(json, is_remote)    
    self.key = json["topology"]["topology"]

  def load(self, worker):
    return load_model_topology_http(self.url, self.key, worker.id_job, worker.username)

