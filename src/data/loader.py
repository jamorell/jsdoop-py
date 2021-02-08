from src.constants import constants as C
from src.data.loader_mnist import loader_mnist


def loader(myjson):
  if myjson["data"]["dataset"].lower() == C.MNIST.lower():
    return loader_mnist(myjson)
