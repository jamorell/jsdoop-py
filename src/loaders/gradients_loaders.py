from src.helper import url_solver as US
from src.api.save_gradients import save_gradients_http

class GradientsLoaderHTTP:
  def __init__(self, json, is_remote):
    self.url = US.get_url_gradients(json, is_remote) 
    

  def save(self, grads_to_save, worker, id_task, start_time):
    names = [weight.name for layer in worker.model.layers for weight in layer.weights]
    names = list(map(lambda x: x[0:x.index(':')], names))
    new_current_age = save_gradients_http(grads_to_save, names, self.url, worker.id_job, worker.age_model, worker.username, id_task, start_time)
    return new_current_age

