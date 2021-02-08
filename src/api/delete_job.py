import requests
from src.helper import url_solver as US

def delete_job(url_delete_job, id_job):
  r = requests.get(url = url_delete_job, params = {"id_job": id_job})
  print(r)

