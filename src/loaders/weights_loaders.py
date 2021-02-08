from src.helper import url_solver as US
from src.api.load_current_weights import load_current_weights_http


class CurrentWeightsLoaderHTTP:
  def __init__(self, json, is_remote):
    self.url = US.get_url_current_weights(json, is_remote) 
    

  def load(self, worker, id_task):
    return int(load_current_weights_http(worker.model, worker.age_model, self.url, worker.id_job, worker.username, id_task))

