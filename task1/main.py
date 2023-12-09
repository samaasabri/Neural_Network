import numpy as np
import tkinter as tk
# import ttkbootstrap as ttk # [Line 2]
from tkinter import ttk
# from Data_Preprocessing import plot_data # [Line 1]
# from train import *
import Data_Preprocessing as pp
from adaline_algo import Adaline
from perceptron_ import perceptron
from testing_ import evaluation, test_fn

# window
root = tk.Tk() # [Line 1]
# root = ttk.Window(themename = "morph") # [Line 2]
root.title("Task1")
root.geometry("1000x500+700+200")
root.configure(background="#D9E3F1")

# functions 


# data

selected_class_option = tk.IntVar()
selected_algo_option = tk.IntVar()
bias_var = tk.BooleanVar(value=False)


# Widgets

## row1

row1 = ttk.Frame(master = root)

class_container = tk.LabelFrame(master = row1, text="Classes")
class1 = tk.Radiobutton(master = class_container, text="C1 & C2", value = 1,  variable = selected_class_option, font="Calibri 13")
class2 = tk.Radiobutton(master = class_container, text="C1 & C3",  value = 2, variable = selected_class_option,font="Calibri 13")
class3 = tk.Radiobutton(master = class_container, text="C2 & C3",  value = 3, variable = selected_class_option,font="Calibri 13")


bias_chk = ttk.Checkbutton(master=row1, text = "Adding Bias", variable=bias_var)

algorithm_container = tk.LabelFrame(master = row1, text="Algorithms")
preceptron_algo = tk.Radiobutton(master = algorithm_container, text="Preceptron", value = 1,  variable = selected_algo_option, font="Calibri 13")
adaline_algo = tk.Radiobutton(master = algorithm_container, text="Adaline",  value = 2, variable = selected_algo_option, font="Calibri 13")


## row2
row2 = ttk.Frame(master = root)


learning_rate_input = ttk.Entry(master = row2)
learning_rate_label = tk.Label(master = row2, text = "Learning Rate (eta)", font = "Calibri 13")

epochs_number_input = ttk.Entry(master = row2)
epochs_number_label = tk.Label(master = row2, text = "Number of epochs (m)", font = "Calibri 13")

## row3

row3 = ttk.Frame(master = root)

mse_threshold_input = ttk.Entry(master = row3)
mse_threshold_label = tk.Label(master = row3, text = "MSE Threshold", font = "Calibri 13")

data = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength" ,"roundnes"]

feature1_combo = ttk.Combobox(master = row3, values = data)
feature1_combo_label = tk.Label(master = row3, text = "Feature 1", font = "Calibri 13")


feature2_combo = ttk.Combobox(master = row3, values = data)
feature2_combo_label = tk.Label(master = row3, text = "Feature 2", font = "Calibri 13")


# layout

## row1

row1.pack(pady=30)

class_container.pack(side="left", padx=10)  
class1.pack()
class2.pack()
class3.pack()

bias_chk.pack(side="left", padx=10)

algorithm_container.pack(side="left", padx=10)  
preceptron_algo.pack()
adaline_algo.pack()

## row2

row2.pack(pady=30)

learning_rate_input.pack(side="left", padx=10)
learning_rate_label.pack(side="left", padx=10)

epochs_number_input.pack(side="left", padx=10)
epochs_number_label.pack(side="left", padx=10)

## row3

row3.pack(pady=30)

mse_threshold_input.pack(side="left", padx=10)
mse_threshold_label.pack(side="left", padx=10)

feature1_combo.pack(side="left", padx=10)
feature1_combo_label.pack(side="left", padx=10) 


feature2_combo.pack(side="left", padx=10)
feature2_combo_label.pack(side="left", padx=10)

# -------------------------------------------- #


# wind2 = ttk.Window(themename = "morph") # [Line 2]

# # take input from gui

def testPredict():
    win2 = tk.Tk()
    win2.title("Task1 Results")
    win2.geometry("1000x500+700+200")
    win2.configure(background="#D9E3F1")
    

    row1 = ttk.Frame(master = win2)

    feature1_input = ttk.Entry(master = row1)
    feature1_label = tk.Label(master = row1, text = "Feature 1", font = "Calibri 13")
    
    feature2_input = ttk.Entry(master = row1)
    feature2_label = tk.Label(master = row1, text = "Feature 2", font = "Calibri 13")
    
    row1.pack(pady=10)

    feature1_input.pack(side="left", padx=10)
    feature1_label.pack(side="left", padx=10)
    
    feature2_input.pack(side="left", padx=20)
    feature2_label.pack(side="left", padx=20)
    
    
    def test_sample():
        data = np.array([])

        data = np.append(data,float(feature1_input.get()))  
        data = np.append(data,float(feature2_input.get()))  
        
        # data = normalization(data)
        
        print("data --> ", data)


        r = test_fn(data, w, b)
        print("test result --> ", r)


    testBtn = ttk.Button(master = win2, text="Test Sample", command=test_sample)
    
    testBtn.pack()
    
    win2.mainloop()

def predict():
    class_choice = selected_class_option.get()
    f1 = feature1_combo.get()
    f2 = feature2_combo.get()
    algo_choice = selected_algo_option.get()
    lr = float(learning_rate_input.get())
    enum = int(epochs_number_input.get())
    
    
    l1 = ["BOMBAY", "CALI", "SIRA"] 
    c1 = ""
    c2 = ""

    if class_choice == 1:
        c1 = l1[0]
        c2 = l1[1]
    elif class_choice == 2:
        c1 = l1[0]
        c2 = l1[2]
    elif class_choice == 3:
        c1 = l1[1]
        c2 = l1[2]

    ms = float(mse_threshold_input.get())

    x_train1, x_test1, y_train1, y_test1 = pp.Split_my_data(choice=class_choice, feature1=f1, feature2=f2)


    global w, b

    if algo_choice == 1:
        w_perceptron,b_perceptron = perceptron(x_train1, y_train1, lr, enum, use_bias=bias_var.get())
        y_perceptron = (test_fn(x_test1, w_perceptron, b_perceptron))
        evaluation(y_perceptron, y_test1, c1,c2)
        pp.plot_data(x_test1, y_test1, w_perceptron, 0,"Perceptron")
        w, b = w_perceptron, b_perceptron
    elif algo_choice == 2:
        w_Adaline,b_Adaline = Adaline(x_train1, y_train1,lr, enum, ms, use_bias=bias_var.get())
        y_Adaline = (test_fn(x_test1, w_Adaline, b_Adaline))
        evaluation(y_Adaline, y_test1,c1,c2)
        pp.plot_data(x_test1, y_test1, w_Adaline, 0, "Adaline")
        w, b =  w_Adaline, b_Adaline
        
    testPredict()
        

predict_btn = ttk.Button(master = root, text = "Predict", command = predict)


predict_btn.pack()

# run
root.mainloop()