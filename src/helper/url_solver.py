from src.constants import constants as C


def get_url_save_job(is_remote):
  path = "/save_job"
  if is_remote:
    return C.JOB_HOST_REMOTE + ":" + str(C.JOB_PORT_REMOTE) + path
  else:
    return C.JOB_HOST_LOCAL + ":" + str(C.JOB_PORT_LOCAL) + path

def get_url_job(is_remote):
  path = "/get_job"
  if is_remote:
    return C.JOB_HOST_REMOTE + ":" + str(C.JOB_PORT_REMOTE) + path
  else:
    return C.JOB_HOST_LOCAL + ":" + str(C.JOB_PORT_LOCAL) + path


def get_url_delete_job(is_remote):
  path = "/delete_job"
  if is_remote:
    return C.JOB_HOST_REMOTE + ":" + str(C.JOB_PORT_REMOTE) + path
  else:
    return C.JOB_HOST_LOCAL + ":" + str(C.JOB_PORT_LOCAL) + path

def get_url_topology(job_json, is_remote):
  the_type = "topology"
  path = "/topology"
  return __generic_url(job_json, the_type, path, is_remote)
  
def get_url_current_weights(job_json, is_remote):
  the_type = "weights"
  path = "/current_weights"
  return __generic_url(job_json, the_type, path, is_remote)

def get_url_gradients(job_json, is_remote):
  the_type = "gradients"
  path = "/gradients"
  return __generic_url(job_json, the_type, path, is_remote)

def get_url_delete_gradients(job_json, is_remote):
  the_type = "gradients"
  path = "/delete_gradients"
  return __generic_url(job_json, the_type, path, is_remote)


def get_url_old_weights(job_json, is_remote):
  the_type = "weights"
  path = "/old_weights"
  return __generic_url(job_json, the_type, path, is_remote)


def __generic_url(job_json, the_type, path, is_remote):
  if is_remote:
    host = "host_remote"
    port = "port_remote"
    return job_json[the_type][host] + ":" + str(job_json[the_type][port]) + path
  else:
    host = "host_local"
    port = "port_local"
    return job_json[the_type][host] + ":" + str(job_json[the_type][port]) + path



 


def get_rabbit_params(job_json):
  return [job_json["aggregator"]["rabbit_host"], job_json["aggregator"]["rabbit_port"] ]




'''
def get_url_job_local():
  return C.JOB_HOST_LOCAL + ":" + str(C.JOB_PORT_LOCAL) + "/get_job"

def get_url_delete_job_local():
  return C.JOB_HOST_LOCAL + ":" + str(C.JOB_PORT_LOCAL) + "/delete_job"

def get_url_job_remote():
  return C.JOB_HOST_REMOTE + ":" + str(C.JOB_PORT_REMOTE) + "/get_job"

def get_url_delete_job_remote():
  return C.JOB_HOST_REMOTE + ":" + str(C.JOB_PORT_REMOTE) + "/delete_job"


def get_url_topology_local(job_json):
  return job_json["topology"]["host"] + ":" + str(job_json["topology"]["port"]) + "/topology"

def get_url_topology_remote(job_json):
  return job_json["topology"]["host"] + ":" + str(job_json["topology"]["port"]) + "/topology"


def get_url_current_weights_local(job_json):
  return job_json["weights"]["host"] + ':' + str(job_json["weights"]["port"]) + '/current_weights'

def get_url_current_weights_remote(job_json):
  return job_json["weights"]["host"] + ':' + str(job_json["weights"]["port"]) + '/current_weights'


def get_url_current_weights_(host, port):
  return host + ':' + str(port) + '/current_weights'


def get_url_gradients_local(job_json):
  return job_json["gradients"]["host"] + ':' + str(job_json["gradients"]["port"]) + '/gradients'

def get_url_gradients_remote(job_json):
  return job_json["gradients"]["host"] + ':' + str(job_json["gradients"]["port"]) + '/gradients'

def get_url_old_weights(job_json):
  if (C.DEBUG):
    job_json["weights"]["host"] = C.JOB_HOST
    job_json["weights"]["port"] = C.JOB_PORT
  return job_json["weights"]["host"] + ':' + str(job_json["weights"]["port"]) + '/old_weights'


def get_rabbit_params(job_json):
  if (C.DEBUG):
    job_json["aggregator"]["rabbit_host"] = "localhost"
    job_json["aggregator"]["rabbit_port"] = 5672
  print(job_json["aggregator"]["rabbit_host"])
  print(job_json["aggregator"]["rabbit_port"])
  return [job_json["aggregator"]["rabbit_host"], job_json["aggregator"]["rabbit_port"] ]
'''


