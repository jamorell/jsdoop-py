# Task Solver -> ts
def is_end_max_global_age(ts): 
  if ts.age_model >= ts.p.global_max_age_model:
    #ts.is_running = False
    ts.finalize()


def is_connected_device(ts):
  if (ts.p.device.is_connected):
    ts.p.device.next_step(time.time() - ts.ss_start_time)
  else:
    print("Disconnected " + str(ts.p.device.remaining_steps) + " seconds!")
    time.sleep(ts.p.device.remaining_steps)
    ts.p.device.next_step(ts.p.device.remaining_steps + 1)
    ts.initialize()    
    print("CONNECTED " + str(ts.p.device.remaining_steps) + " seconds!")


