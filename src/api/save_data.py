import requests
import io
from tempfile import SpooledTemporaryFile
import numpy as np

def save_data_http(mylist, url_data_server, key):
  #outfile = SpooledTemporaryFile()
  #np.save(outfile, mylist, allow_pickle = True)
  #_ = outfile.seek(0)

  PARAMS = {'key': key}; 
  r = requests.post(url = url_data_server,
                      data = np.array(mylist).tobytes(),#outfile,
                      params = PARAMS,
                      headers={'content-type': 'application/octet-stream'})
  print(r.status_code)
  print(r.content)

