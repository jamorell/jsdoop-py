import json
import requests
import sys

from src.constants import constants as C
from src.constants import job as J



r = requests.post(C.JOB_HOST + ":" + str(C.PORT) + "/save_job", json=J.DEFAULT_JOB) 
#r = requests.get(host + ":" + str(port) + "/get_job")
#r = requests.get("http://localhost:8081/get_job?id_job=3")

print(r)
print(r.content[0:200])
print(r.headers)
print(r.headers["id_job"])
myjob["id_job"] = r.headers["id_job"]

