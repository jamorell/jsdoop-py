from src.helper import url_solver as US
from src.api.load_job import load_job


class JobLoaderHTTP:
  def __init__(self, is_remote):
    self.url = US.get_url_job(is_remote) 
    
  def load(self, id_job):
    print("ur_job " + self.url)
    return load_job(self.url, id_job)	

'''
class JobLoaderJSON:
  def __init__(self, url_job, id_job):
    self.url_job = url_job
    self.id_job = id_job

  def load(self):
    return load_job(self.url_job, self.id_job)	




host = "localhost"
port = 8081
url_job = "http://" + host + ":" + str(port) + "/get_job"
id_job = 1606147964029
job_loader = JobLoaderJSON(url_job, id_job)
print(job_loader.load())

'''
