import numpy as np


#X_train Y_train
def create_batches(Xs, Ys, myjson):
  batchsX = []
  batchsY = []
  listX = []
  listY = []
  counter = 0
  for index in range(Xs.shape[0]):
    #if (counter >= 0 and counter < 1):
    #print("AAshape = " + str(Ys[index].shape))
    print("AAdtype = " + str(Ys[index].dtype))
    listX.append(Xs[index])
    listY.append(Ys[index])
    if ((index + 1) % myjson["data"]["mb_size"] == 0) : 
      xKey = "mnist_" + str(myjson["data"]["mb_size"]) + "_" + str(counter) + "_x.npy"
      print("xKey " + xKey)

      #requestPostData(xKey, np.array(listX).tobytes())
      batchsX.append(np.array(listX))
      yKey = "mnist_" + str(myjson["data"]["mb_size"]) + "_" + str(counter) + "_y.npy"
      print("yKey " + yKey)
      print("shape = " + str(np.array(listY)))
      #requestPostData(yKey, np.array(listY).tobytes())      
      batchsY.append(np.array(listY))
      counter = counter + 1
      listX = []
      listY = []
  return batchsX, batchsY


def get_total_m_batches(myjson):
  return round(myjson["data"]["train_dataset_len"] / myjson["data"]["mb_size"])
  
