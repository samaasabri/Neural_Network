# -*- coding: utf-8 -*-
"""Update_weights.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TMMOnBtW0jPKXa9DZNeplFpfQAWmm2J5
"""

import numpy as np

#Both parameters taken from GUI
#ex >>(3,[10,10,10])

def Intialize_weights(hidden_layers,neurons):
  wieght_matrices=[]
  for i in range(hidden_layers+1):
    w_i=[]
    if i==0:
      w_i=np.random.rand(neurons[i],6)
    elif i== hidden_layers:
      w_i=np.random.rand(3,neurons[i-1])
    else:
      w_i=np.random.rand(neurons[i],neurons[i-1])


    #print(w_i.shape)
    #print('----------------------------------------')
    wieght_matrices.append(w_i)

  return  wieght_matrices

#Taken from GUI
hidden_layers=3
neurons=[10,10,10]

wieght_matrices=Intialize_weights(hidden_layers,neurons)
#print(wieght_matrices)