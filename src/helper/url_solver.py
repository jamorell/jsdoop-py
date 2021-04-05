from src.constants import constants as C


def get_url_data(job_json, is_remote):
  the_type = "data"
  path = "/dataset"
  return __generic_url(job_json, the_type, path, is_remote)

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



