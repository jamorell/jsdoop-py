import os
import signal
import subprocess
from subprocess import call 
import sys
import time
import random

if (len(sys.argv) == 2):
  sys.argv.append("default_worker_name") 

if (len(sys.argv) == 3):
  sys.argv.append("false") 

from src.algorithms.termination_criteria.device_simulator import DeviceSimulator 

var_lambda_when_connect = 6 #400#40 #93.333 #40
var_lambda_when_disconnect = 6 #400#26.666 #160 #93.333 #40 #93.333#80
total_steps = 3600 # 1 hour is the total time (seconds)
connected = True
device = DeviceSimulator(connected, var_lambda_when_connect, var_lambda_when_disconnect, total_steps)
if (random.random() < 0.5):
  device.is_connected = False
else:
  device.is_connected = True  
#with open('./logfile.txt', 'w') as f:
#   call(['sh', './worker.sh', '-m', sys.argv[1], sys.argv[2], sys.argv[3]], stdout=f)
#proc = call(['sh', './worker.sh', sys.argv[1], sys.argv[2], sys.argv[3]])
proc = None

time_to_load_data = 0#40

#FNULL = open(os.devnull, 'w')
print("Starting")
while (True):
  ss_start_time = time.time()
  if (device.is_connected):
    print("Connected " + str(device.remaining_steps))
    if (proc is None):
      proc = subprocess.Popen('sh ./worker.sh ' + sys.argv[1] + ' ' + sys.argv[2] + ' ' + sys.argv[3], shell = True, stdout = subprocess.DEVNULL)
      #proc = call(['sh', './worker.sh', sys.argv[1], sys.argv[2], sys.argv[3]], stdout = FNULL, shell = False)
      print(proc.pid)
      print(os.getpid())
      print(device.remaining_steps)
      device.remaining_steps = device.remaining_steps + 40
      time.sleep(device.remaining_steps)
  else:
    print("Disconnected " + str(device.remaining_steps))
    if (proc is not None):
      print("Killing")
      #os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
      proc.terminate()
      proc = None
      device.remaining_steps = device.remaining_steps - 40
      if device.remaining_steps > 0:
        time.sleep(device.remaining_steps)
  #device.next_step(time.time() - ss_start_time)
  device.remaining_steps = 0
  device.next_step(0)
      



