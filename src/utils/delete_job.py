import requests
import sys

from src.api.delete_job import delete_job
from src.helper import url_solver as US

id_job = None
try:
  id_job = int(sys.argv[1]) #1234567890
except:
  exit("ERROR: Please insert a valid numeric job id. "  + sys.argv[1])

is_remote = False
try:
  is_remote = (sys.argv[2].lower() == 'true')
  print("REMOTE HOST")
except:
  print("LOCAL HOST")



#requests.get(url = job_host + ":" + str(job_port) + "/delete_job", params = {"id_job": id_job})
print(is_remote)
url_delete_job = US.get_url_delete_job(is_remote)
print(url_delete_job)
delete_job(url_delete_job, id_job)
