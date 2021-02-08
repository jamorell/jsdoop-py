import numpy

# We simulate connections and disconnetions of volunteers using exponential distributions.
class DeviceSimulator:
  remaining_steps = 0
  connected_times = []
  disconnected_times = []

  def __init__(self, connected, lc, ld, max_steps):
    self.is_connected = connected
    self.lambda_when_connect = lc
    self.lambda_when_disconnect = ld
    self.expected_max_steps = max_steps
    self.__calculate_steps()

  def __calculate_steps(self):
    if self.is_connected:
      self.remaining_steps = round(numpy.random.exponential(1 / self.lambda_when_connect) * self.expected_max_steps)
      self.connected_times.append(self.remaining_steps)
    else:
      self.remaining_steps = round(numpy.random.exponential(1 / self.lambda_when_disconnect) * self.expected_max_steps)
      self.disconnected_times.append(self.remaining_steps)

  def next_step(self, steps):
    self.remaining_steps = self.remaining_steps - steps
    if (self.remaining_steps <= 0):
      self.is_connected = not self.is_connected
      self.__calculate_steps()

    '''
  def next_step(self):
    self.remaining_steps = self.remaining_steps - 1
    if (self.remaining_steps <= 0):
      self.is_connected = not self.is_connected
      self.__calculate_steps()
'''
