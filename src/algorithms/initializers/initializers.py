# Task Solver -> ts
def init_device(ts): 
  if (ts.p.device.is_connected):
    ts.initialize()
