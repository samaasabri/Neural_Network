from Data_Preprocessing import Split_my_data,plot_data
from adaline_algo import Adaline
from perceptron_ import perceptron
from testing_ import test_fn,evaluation

# spilt Data For C1,C2

choise = 1
x_train1, x_test1, y_train1, y_test1 = Split_my_data(choice=choise, feature1="Perimeter", feature2="MajorAxisLength")

w_Adaline,b_Adaline = Adaline(x_train1, y_train1, 0.01, 1000)

# w_perceptron,b_perceptron = perceptron(x_train1, y_train1, 0.001, 1000)




y_Adaline = (test_fn(x_test1, w_Adaline, b_Adaline))
# y_perceptron = (test_fn(x_test1, w_perceptron, b_perceptron))

evaluation(y_Adaline, y_test1, "BOMBAY", "CALI")
# evaluation(y_perceptron, y_test1, "BOMBAY", "CALI")

plot_data(x_test1, y_test1, w_Adaline, 0, "Adaline")
# plot_data(x_test1, y_test1, w_perceptron, 0,"Perceptron")


