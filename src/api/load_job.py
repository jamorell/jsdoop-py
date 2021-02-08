import json
import requests

def load_job(url, id_job):
  try:
    r = requests.get(url + "?id_job=" + str(id_job))
    print(r)
    print(r.content[0:201])
    print(r.headers)
    myjson = json.loads(r.content.decode('utf-8'))
    return myjson
  except Exception as e:
    print("ERROR " + str(e))
    return None
