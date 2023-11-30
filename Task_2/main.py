import numpy as np
import tkinter as tk
import ttkbootstrap as ttk # [Line 2]
# from tkinter import ttk
# from Data_Preprocessing import plot_data # [Line 1]
import nn2

# window
# root = tk.Tk() # [Line 1]
root = ttk.Window(themename = "morph") # [Line 2]
root.title("Task1")
root.geometry("1000x500+700+200")
root.configure(background="#D9E3F1")

# functions 

def handle_selection():
    print(act_func_option.get())
    
def test_bias():
    print(bias_var.get())
    

# data

act_func_option = tk.IntVar()
bias_var = tk.BooleanVar(value=False)


# Widgets

## row1

row1 = ttk.Frame(master = root)

bias_chk = ttk.Checkbutton(master=row1, text = "Adding Bias", variable=bias_var, command=test_bias)

act_functions_container = tk.LabelFrame(master = row1, text="Activation Functions")

tangent_sigmoid_func  = tk.Radiobutton(master = act_functions_container, text="Hyperbolic Tangent sigmoid",  value = 0, variable = act_func_option, command = handle_selection, font="Calibri 13")

sigmoid_func = tk.Radiobutton(master = act_functions_container, text="Sigmoid", value = 1,  variable = act_func_option, command = handle_selection, font="Calibri 13")


## row2

row2 = ttk.Frame(master = root)


learning_rate_input = ttk.Entry(master = row2)
learning_rate_label = tk.Label(master = row2, text = "Learning Rate (eta)", font = "Calibri 13")

epochs_number_input = ttk.Entry(master = row2)
epochs_number_label = tk.Label(master = row2, text = "Number of epochs (m)", font = "Calibri 13")

## row3

row3 = ttk.Frame(master = root)

hLayers_num_input = ttk.Entry(master = row3)
hLayers_num_label = tk.Label(master = row3, text = "number of hidden layers", font="Calibri 13")

neurons_num_input = ttk.Entry(master = row3)
neurons_num_label = tk.Label(master = row3, text = "number of neurons in each hidden layer", font="Calibri 13")


# layout

## row1

row1.pack(pady=30)

bias_chk.pack(side="left", padx=10)

act_functions_container.pack(side="left", padx=100)  
sigmoid_func.pack()
tangent_sigmoid_func.pack()

## row2

row2.pack(pady=30)

learning_rate_input.pack(side="left", padx=10)
learning_rate_label.pack(side="left", padx=10)

epochs_number_input.pack(side="left", padx=10)
epochs_number_label.pack(side="left", padx=10)

## row3

row3.pack(pady=30)

hLayers_num_input.pack(side="left", padx=10)
hLayers_num_label.pack(side="left", padx=10)

neurons_num_input.pack(side="left", padx=10)
neurons_num_label.pack(side="left", padx=10)

# -------------------------------------------- #

# take input from gui


def predict_window():

    win2 = tk.Tk()
    win2.title("Task2 Prediction")
    win2.geometry("1000x500+700+200")
    win2.configure(background="#D9E3F1")
    
    
    row1 = ttk.Frame(master = win2)


    global predict_input
    predict_input = []

    area_input = ttk.Entry(master = row1)
    area_label = tk.Label(master = row1, text = "Area", font = "Calibri 13")
    
    
    permieter_input = ttk.Entry(master = row1)
    permieter_label = tk.Label(master = row1, text = "Perimeter", font = "Calibri 13")
    
    row1.pack(pady=10)

    area_input.pack()
    area_label.pack()

    permieter_input.pack()
    permieter_label.pack()

    row2 = ttk.Frame(master = win2)

    major_axis_len_input = ttk.Entry(master = row1)
    major_axis_len_label = tk.Label(master = row1, text = "MajorAxisLength", font = "Calibri 13")

    minor_axis_len_input = ttk.Entry(master = row1)
    minor_axis_len_label = tk.Label(master = row1, text = "MinorAxisLength", font = "Calibri 13")
    
    
    roundnes_input = ttk.Entry(master = row1)
    roundnes_label = tk.Label(master = row1, text = "Roundnes", font = "Calibri 13")
    
    
    row2.pack(pady=10)
    
    major_axis_len_input.pack()
    major_axis_len_label.pack()
    
    minor_axis_len_input.pack()
    minor_axis_len_label.pack()
    
    roundnes_input.pack()
    roundnes_label.pack()
    

    global get_predict_input

    def get_predict_input():
        predict_input.append(float(area_input.get()))
        predict_input.append(float(permieter_input.get()))
        predict_input.append(float(major_axis_len_input.get()))
        predict_input.append(float(minor_axis_len_input.get()))
        predict_input.append(float(roundnes_input.get()))
        predict_input.append(1.0)


    predictionBtn = ttk.Button(master = win2, text="Predict", command=perform_prediction)

    predictionBtn.pack()

    
    win2.mainloop()

def perform_prediction():

    get_predict_input()
    tmp = predict_input.copy()
    tmp2 = np.array(tmp).reshape(1, 6)

    prediction = nn2.predict(tmp2, neurons_num, net, act_func_option.get(), hidden_layers_num)
    
    print("[Prediction]: ", prediction)
    

def perform_classification():
    global neurons_num
    neurons_num = neurons_num_input.get().split(',')
    neurons_num = [int(num) for num in neurons_num]
    
    global hidden_layers_num
    hidden_layers_num = int(hLayers_num_input.get())
    
    # 0 -> tanjenet
    # 1 -> sigmoid
    
    learn_rate = float(learning_rate_input.get())
    
    epochs_num = int(epochs_number_input.get())
    
    global net
    net = nn2.train(nn2.x_train, nn2.y_train, hidden_layers_num, neurons_num, act_func_option.get(), learn_rate, epochs_num)
    
    
    y_train_indeices = []
    for i in nn2.y_train:
        for j in i:
            for k in range(len(j)):
                if j[k] == 1:
                    y_train_indeices.append(k)
    
    prediction = nn2.predict(nn2.x_train, neurons_num, net, act_func_option.get(), hidden_layers_num)
    print("[Prediction]: ", prediction)
    print("[y_train]: ", y_train_indeices)
    
    
    y_test_indeices = []
    for i in nn2.y_test:
        for j in i:
            for k in range(len(j)):
                if j[k] == 1:
                    y_test_indeices.append(k)
    
    prediction = nn2.predict(nn2.x_test, neurons_num, net, act_func_option.get(), hidden_layers_num)
    print("[Prediction]: ", prediction)
    print("[y_test]: ", y_test_indeices)
    
        
    predict_window()


classifyBtn = ttk.Button(master = root, text="Classify", command=perform_classification)

classifyBtn.pack()

# run
root.mainloop()