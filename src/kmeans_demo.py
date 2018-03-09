import matplotlib.pyplot as plt
from network.csv_loader import load_csv
from network.kmc import k_means

#load training data and read out inputs
training_data = load_csv("res/dataset.csv", 2, 1)
training_inputs = [d[0] for d in training_data]

#scatter plot all training points
plt.scatter([inp[0] for inp in training_inputs], [inp[1] for inp in training_inputs])

#find and scatter plot all means
means = k_means(training_inputs, 20, 25)
plt.scatter([inp[0] for inp in means], [inp[1] for inp in means], marker="^")

plt.show()
