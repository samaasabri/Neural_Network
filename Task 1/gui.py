import tkinter as tk
import ttkbootstrap as ttk # [Line 2]
# from tkinter import ttk # [Line 1]

# window
# root = tk.Tk() # [Line 1]
root = ttk.Window(themename = "morph") # [Line 2]
root.title("Task1")
root.geometry("1000x500+700+200")
root.configure(background="#D9E3F1")

# functions 

def handle_selection():
    print(selected_class_option.get())
    
def test_bias():
    print(bias_var.get())
    

# data


selected_class_option = tk.IntVar()
selected_algo_option = tk.StringVar()
bias_var = tk.BooleanVar(value=False)


# Widgets

## row1

row1 = ttk.Frame(master = root)

class_container = tk.LabelFrame(master = row1, text="Classes")
class1 = tk.Radiobutton(master = class_container, text="C1 & C2 ", value = 1,  variable = selected_class_option, command = handle_selection, font="Calibri 13")
class2 = tk.Radiobutton(master = class_container, text="C1 & C3",  value = 2, variable = selected_class_option, command = handle_selection, font="Calibri 13")
class3 = tk.Radiobutton(master = class_container, text="C2 & C3",  value = 3, variable = selected_class_option, command = handle_selection, font="Calibri 13")


bias_chk = ttk.Checkbutton(master=row1, text = "Adding Bias", variable=bias_var, command=test_bias)

algorithm_container = tk.LabelFrame(master = row1, text="Algorithms")
preceptron_algo = tk.Radiobutton(master = algorithm_container, text="Preceptron", value = 1,  variable = selected_algo_option, command = handle_selection, font="Calibri 13")
adaline_algo = tk.Radiobutton(master = algorithm_container, text="Adaline",  value = 2, variable = selected_algo_option, command = handle_selection, font="Calibri 13")


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



# run
root.mainloop()