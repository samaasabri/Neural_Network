import numpy as np

Input_layer=[1,1,1,1,1,1]
hidden_layers=3
neurons=[10,10,10]
activision=1
Target=[0,0,1]


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


    wieght_matrices.append(w_i)

  return  wieght_matrices

wieght_matrices=Intialize_weights(hidden_layers,neurons)


def feedForward(Input_layer,neuron_num,wieght_matrices,activision, hidden_layers):
    #activison = 1 ---> sigmoid
    #activison = 0 ---> tengent
  
  neurons = neuron_num.copy()
  neurons.append(3)
  #contains all f for each layer
  array_of_fs=[]

  def sigmoid(f_i):
    return 1/(1+np.exp(-f_i))

  def tengent(f_i):
    return np.tanh(f_i)

  for i in range(hidden_layers+1):

    #f contains net value for wach neuron
    f=[]

    for n in range(neurons[i]):

      if i==0:
        f_i=np.dot(wieght_matrices[i][n],Input_layer)
        if activision:
            f_i = sigmoid(f_i)
        else:
            f_i = tengent(f_i)
        f.append(f_i)
      else:
        f_i=np.dot(wieght_matrices[i][n],array_of_fs[i-1])
        if activision:
            f_i = sigmoid(f_i)
        else:
            f_i = tengent(f_i)
        f.append(f_i)


    array_of_fs.append(f)
  return  array_of_fs

def calc_weight_arr(arr_of_arrs,idx):
  arr_of_wights=[]
  for i in arr_of_arrs:
    arr_of_wights.append(i[idx])
  return arr_of_wights

def sigmoid_dash(f_i):
    return ((f_i)*(1-f_i))

def tengent_dash(f_i):
    return (1-pow(np.tanh(f_i),2))

def backPropagate(Target,neuron_num,wieght_matrices,activision,array_of_fs, hidden_layers):

  neurons = neuron_num.copy()
  neurons.append(3)

  array_of_errors=[]
  j=-1

  for i in range(hidden_layers,-1,-1):
    # print("Layer:" ,i)
    error=[]

    for n in range(neurons[i]):
      # print("Neuron: ",n)
    #check activision function
      if activision:
        dash = sigmoid_dash(array_of_fs[i][n])

        if i==hidden_layers:
          error_i=(Target[n]- array_of_fs[i][n])*dash

        else:
          wights=calc_weight_arr(wieght_matrices[i+1],n)

          error_i= np.dot(wights,array_of_errors[j]) *dash


      else:
          error_i = tengent_dash(array_of_fs[i][n])

      error.append(error_i)
    array_of_errors.append(error)
    j+=1
  return array_of_errors


def Update_wieghts(array_of_errors,wieght_matrices,Input_layer,learning_rate,hidden_layers,neuron_num,array_of_fs):


  neurons = neuron_num.copy()
  neurons.append(3)

  for i in range(hidden_layers+1):
    #print("Layer: ",i)


    for n in range(neurons[i]):
      #print("Neuron: ",n)
      #print(wieght_matrices[i][n])
      for j in range(len(wieght_matrices[i][n])):


         if i==0:
           wieght_matrices[i][n][j]=wieght_matrices[i][n][j]+learning_rate*array_of_errors[i][n]*Input_layer[j]

         else:
           wieght_matrices[i][n][j]=wieght_matrices[i][n][j]+learning_rate*array_of_errors[i][n]*array_of_fs[i][n]


  return wieght_matrices


def train(Input,target,hidden_layers_num,neurons_num,Activation_used,learninR,num_epochs):

  net=Intialize_weights(hidden_layers=hidden_layers_num,neurons=neurons_num)

  for i in range(num_epochs):
    for j in range(len(Input)):
      FF=feedForward(Input[j],neurons_num,net,Activation_used, hidden_layers_num)
      array_of_error=backPropagate(target[j][0],neurons_num,net,Activation_used,FF, hidden_layers_num)
      array_of_error.reverse()
      net=Update_wieghts(array_of_error,net,Input[j],learninR,hidden_layers_num,neurons_num,FF)


  return net

def predict(input,neurons_num,net,Activation_used, hidden_layers):
  predictions=[]
  for sample in input:
    FF=feedForward(sample,neurons_num,net,Activation_used, hidden_layers)
    # print(FF[-1])
    predictions.append(np.argmax(FF[-1]))

  return predictions

import pandas as pd
import numpy as np

data = pd.read_excel("./Data/Dry_Bean_Dataset.xlsx")

data['MinorAxisLength'].fillna(value=data['MinorAxisLength'].mean(), inplace=True)

data.info()

def encode(x):
  res=np.zeros((1,3))
  if x=='BOMBAY':
    res[0][0]=1
  elif x=='CALI':
    res[0][1]=1
  else:
    res[0][2]=1

  return res.tolist()

def normalization(data):

  data = (data - data.min()) / (data.max() - data.min()) * (10-1) - 1
  print(str(data.min())+" "+str( data.max()))
  return data, data.min(), data.max()


def normalization_test(data,dMin,dMax):
  data = (data - dMin) / (dMax - dMin) * (10-1) - 1
  print(str(dMin)+" "+str(dMax))
  return data

from sklearn.preprocessing import StandardScaler

def preprocess(data):

    scaler = StandardScaler()

    d1 = data.iloc[:30, :]
    d2 = data.iloc[50:80, :]
    d3 = data.iloc[100:130,:]
    d = pd.concat([d1, d2])
    df= pd.concat([d, d3])
    df=df.sample(frac=1)
    #   x_train,DMIN,DMAX=normalization(df.iloc[:,:-1])
    df1 = df.iloc[:,:-1]
    df1["X1"] = 1
    model = scaler.fit(df1)
    df1 = model.transform(df1)
    x_train=df1.tolist()

    y_train=[]
    for i in df['Class']:
        y_train.append(encode(i))

    d1 = data.iloc[30:50, :]
    d2 = data.iloc[80:100, :]
    d3 = data.iloc[130:,:]
    d = pd.concat([d1, d2])
    df= pd.concat([d, d3])
    df=df.sample(frac=1)

    df2 = df.iloc[:,:-1]
    df2["X1"] = 1
    df2 = model.transform(df2)
    x_test=df2.tolist()
    y_test=[]
    for i in df['Class']:
        y_test.append(encode(i))

    return x_train,x_test,y_train,y_test




x_train,x_test,y_train,y_test = preprocess(data)

# neurons = [12,6,3]
# net = train(x_train,y_train,3,neurons,1,0.01,100)

# preds=predict(x_test,neurons,net,1)

# preds

# from sklearn.metrics import confusion_matrix

# # confusion_matrix funnction a matrix containing the summary of predictions
# print(confusion_matrix(y_test, predictions))