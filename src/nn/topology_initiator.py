import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D


from src.constants import constants as C
from src.nn.topology_mnist_conv_28_28_1 import create_topology_mnist_conv_28_28_1

def create_topology(topology_name):
  if(topology_name.lower() == C.MNIST_CONV_28_28_1.lower()):
    return create_topology_mnist_conv_28_28_1()
  else:
    return None
  #if(job_json["topology"]["topology"].lower() == C.MOBILENET_224_224_3.lower()):    
  #   return topology_mnist_conv_28_28_1()   

