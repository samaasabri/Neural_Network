from Data_Preprocessing import Split_my_data,plot_data
from adaline_algo import Adaline
from perceptron_ import perceptron
from testing_ import test_fn,evaluation
import gui

# take input from gui

class_choice = 1

if gui.selected_class_option == 2:
    class_choice = 2
elif gui.selected_class_option == 3:
    class_choice = 3

algo_choice = 1

if gui.selected_algo_option == 2:
    algo_choice = 2


f1 = gui.feature1_combo.get()
f2 = gui.feature2_combo.get()


lr = float(gui.learning_rate_input.get())
epochs_num = float(gui.epochs_number_input.get())
mse = float(gui.mse_threshold_input.get())


# spilt Data For C1,C2

x_train1, x_test1, y_train1, y_test1 = Split_my_data(choice=class_choice, feature1=f1, feature2=f2)

if algo_choice == 1:
    w_Adaline,b_Adaline = Adaline(x_train1, y_train1, learning_rate=lr,epochs=epochs_num, mse_threshold=mse, use_bias=gui.bias_var.get())
    y_Adaline = (test_fn(x_test1, w_Adaline, b_Adaline))
    evaluation(y_Adaline, y_test1, "BOMBAY", "CALI")
    plot_data(x_test1, y_test1, w_Adaline, 0, "Adaline")

elif algo_choice == 2:
    w_perceptron,b_perceptron = perceptron(x_train1, y_train1, learning_rate=lr,epochs=epochs_num, mse_threshold=mse, use_bias=gui.bias_var.get())
    y_perceptron = (test_fn(x_test1, w_perceptron, b_perceptron))
    evaluation(y_perceptron, y_test1, "BOMBAY", "CALI")
    plot_data(x_test1, y_test1, w_perceptron, 0,"Perceptron")
    
    




