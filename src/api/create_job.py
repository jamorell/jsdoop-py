import json
import requests
import sys

from src.constants import constants as C

def create_job(myjob, url):
#  r = requests.post(C.JOB_HOST + ":" + str(C.JOB_PORT) + "/save_job", json=myjob)
  print(url)
  r = requests.post(url, json=myjob)  
  return r.headers["id_job"]
  

