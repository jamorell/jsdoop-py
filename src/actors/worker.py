import sys
import random 
import time
import tensorflow as tf


from src.constants import constants as C
from src.api.load_job import load_job
from src.data.create_batches import create_batches, get_total_m_batches
from src.helper import url_solver as US

from src.loaders.job_loaders import JobLoaderHTTP
from src.loaders.topology_loaders import TopologyLoaderHTTP
from src.loaders.weights_loaders import CurrentWeightsLoaderHTTP
from src.loaders.gradients_loaders import GradientsLoaderHTTP
from src.loaders.loss_loaders import LossLoaderJSON
from src.loaders.dataset_loaders import DatasetLoaderNonIID #DatasetLoaderHTTP #DatasetLoaderJSON

from src.algorithms.termination_criteria.device_simulator import DeviceSimulator 
import src.algorithms.termination_criteria.termination_criteria as tc #is_end_max_global_age
import src.algorithms.initializers.initializers as inits


#class ProblemStateDistNN:

def mystart(device): 
  device.initialize()

class ProblemJSONDistNN:
  pre_init = []
  post_init = []
  pre_run = []  
  post_run = []

  def __init__(self, json, is_remote, seed): 

    self.json = json

    # LOAD MODEL TOPOLOGY
    print("self.json = " + str(self.json))
    self.topology_loader = TopologyLoaderHTTP(self.json, is_remote)


    ### LOAD MODEL CURRENT WEIGHTS
    self.current_weights_loader = CurrentWeightsLoaderHTTP(self.json, is_remote)

    ### INIT GRADIENTS LOADER
    self.gradients_loader = GradientsLoaderHTTP(self.json, is_remote)

    ### INIT LOSS FUNCTION
    self.loss_calculator = LossLoaderJSON(self.json)

    ### INIT LOADING DATA
    self.dataset = DatasetLoaderNonIID(self.json, seed, is_remote) #DatasetLoaderHTTP(self.json, seed, is_remote) #DatasetLoaderJSON(self.json, seed)

    ### TERMINATION CRITERIA
    try:
      self.global_max_age_model = self.json["termination_criteria"]["global_max_age_model"]
      self.post_run.append(tc.is_end_max_global_age)
    except:
      print("No termination criteriaaa")
    #  self.global_max_age_model = 99999999

    ###### Device Simulation
    '''
    var_lambda_when_connect = 40 #93.333 #40
    var_lambda_when_disconnect = 26.666 #160 #93.333 #40 #93.333#80
    total_steps = 3600 # 1 hour is the total time (seconds)
    self.device = DeviceSimulator(True, var_lambda_when_connect, var_lambda_when_disconnect, total_steps)
    self.post_init.append(inits.init_device)
    '''
    self.post_init.append(mystart)
    ### INIT LOCAL COMPUTATION
    if self.json["worker"]["local_computation"]:
      self.local_computation = True
      self.local_steps = self.json["worker"]["local_steps"]
      #optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
      self.optimizer = tf.keras.optimizers.deserialize(self.json["optimizer"])
      print(tf.keras.optimizers.serialize(self.optimizer))
      print("self.local_steps = " + str(self.local_steps))
    else:
    ### INIT WITHOUT LOCAL COMPUTATION
      self.local_computation = False


#Task Solver
class TaskSolverDistNN:

  ss_start_time = 0 

  def __init__(self, id_job, username, problem): 
    self.id_job = id_job
    self.username = username
    self.p = problem #problem solver
    self.is_running = True
    for posti in self.p.post_init:
      posti(self)


  def initialize(self):
    # LOAD MODEL TOPOLOGY
    self.model = self.p.topology_loader.load(self)
    self.model.summary()
    
    ### LOAD MODEL CURRENT WEIGHTS
    self.age_model = -1
    self.age_model = self.p.current_weights_loader.load(self, 0)

    ### LOCAL COMPUTATION 
    if self.p.local_computation:
      self.listOfGrads = []
      self.finalGrads = []    


  def local_computation(self, id_task, start_time):
    xs, ys, mb_selected = self.p.dataset.get_random_batch(self, id_task, start_time)
    loss_value, grads = self.p.loss_calculator.grad(self.model, xs, ys)
    print("---> LOSS = " +  str(loss_value))
    #''' INIT LOCAL COMPUTATION '''
    self.p.optimizer.apply_gradients(zip(grads, self.model.trainable_variables), experimental_aggregate_gradients=True) 
    for i in range(len(grads)):
      while (i >= len(self.listOfGrads)):
        self.listOfGrads.append([]) 
      self.listOfGrads[i].append(grads[i])
    #''' END LOCAL COMPUTATION '''
    #if (len(listOfGrads[0]) >= self.p.local_steps):
    if (len(self.listOfGrads[0]) >= self.p.local_steps): 
      ''' INIT LOCAL COMPUTATION '''
      for i in range(len(self.listOfGrads)):
        self.finalGrads.append(tf.math.add_n(self.listOfGrads[i]))
      #optimizer.apply_gradients(zip(finalGrads, mnist_model.trainable_variables))
      new_current_age = self.p.gradients_loader.save(self.finalGrads, self, id_task, start_time)
      self.listOfGrads.clear()
      self.finalGrads.clear()
      ''' END LOCAL COMPUTATION '''

      print("new_current_age " + str(new_current_age))
      print("self.age_model " + str(self.age_model))
      self.update_model(new_current_age, id_task) 
       

  def without_local_computation(self, id_task, start_time):
    xs, ys, mb_selected = self.p.dataset.get_random_batch(self, id_task, start_time)
    loss_value, grads = self.p.loss_calculator.grad(self.model, xs, ys)
    print("---> LOSS = " +  str(loss_value))
    new_current_age = self.p.gradients_loader.save(grads, self, id_task, start_time)
    self.update_model(new_current_age, id_task)


  def update_model(self, new_current_age, id_task):
    if (int(new_current_age) > int(self.age_model)):        
      self.age_model = self.p.current_weights_loader.load(self, id_task)
      print("self.age_model = " + str(self.age_model))

  def run(self):
    while True:
      if self.is_running:
        self.ss_start_time = time.time()
        ms_start_time = int(round(time.time() * 1000))
        print("time.time() = " + str(time.time()))
        print("ms_start_time = " + str(ms_start_time))
        id_task = ms_start_time
        #while not check_termination_criterion():

        if self.p.local_computation:
          self.local_computation(id_task, ms_start_time)
        else:
          self.without_local_computation(id_task, ms_start_time)

      #Check Termination
      for postr in self.p.post_run:
        postr(self)


  def finalize(self):
    print("Finishing ...")
    exit()

print("str(argv) = " + str(sys.argv))

id_job = None
try:
  id_job = int(sys.argv[1]) #1606147964029
except:
  exit("ERROR: Please insert a valid numeric job id.")

username = None
try:
  username = "worker_py_" + str(sys.argv[2])
except:
  exit("ERROR: Please insert a valid username.")

seed = None
try:
  seed = int(sys.argv[3])
except:
  exit("ERROR: Please insert a valid seed.")


is_remote = False
try:
  is_remote = (sys.argv[4].lower() == 'true')
  print("REMOTE HOST")
except:
  print("LOCAL HOST")

print("len(argv) = " + str(len(sys.argv)))
print("id_job = " + str(id_job))
print ("username = " + username)
print("is_remote = " + str(is_remote))




# LOAD JOB
job_loader = JobLoaderHTTP(is_remote)
json = job_loader.load(id_job)
print("json = " + str(json))

problem = ProblemJSONDistNN(json, is_remote, seed)
worker = TaskSolverDistNN(id_job, username, problem)
worker.run()
