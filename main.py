
import ANN as N
from numpy.random import seed
from numpy import arange, mean
import matplotlib.pylab as plt

seed(1)

Network = N.GeneralNetwork(3, [2, 3, 2], verbose = 1, vis = False, \
                         activation_function = "sigmoid")
# define network architecture using GeneralNetwork object
Data_obj    = N.Data(10, highamdata = True)
# create data using Data object
cost1   = Network.train(Data_obj)


av_cost1 = [mean(cost1[i, :]) for i in arange(cost1.shape[0])]
# train the network with the data

Network = N.GeneralNetwork(3, [2, 20, 2], verbose = 1, vis = True, \
                         activation_function = "sigmoid")
# define network architecture using GeneralNetwork object
Data    = N.Data(10, highamdata = True)
# create data using Data object
cost2 = Network.train(Data, replacement = 1)
# train the network with the data
av_cost2 = [mean(cost2[i, :]) for i in arange(cost2.shape[0])]

import matplotlib.pylab as plt
# import plotting module
plt.plot(av_cost1)
plt.plot(av_cost2)
plt.legend(['Without Replacement', 'With Replacement'])
plt.xlabel(r'Epoch', fontsize = 20)
plt.ylabel(r'Cost', fontsize = 20)
plt.savefig('demonstration.png')
plt.show()