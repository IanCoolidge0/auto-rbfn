import matplotlib.pyplot as plt
from network.csv_loader import load_csv
from network.rbfn import RBFN
import numpy as np

training_data = load_csv("res/dataset.csv", 2, 1)
training_inputs = map(lambda x: x[0], training_data)

network = RBFN(2, 30)
network.gen_centers(training_data, 10)

network.pinv_train(training_data)
network.evaluate(training_data)

e1 = []
e2 = []

for example in training_data:
    if network.feedforward(example[0]) < 1.5:
        e1.append(example)
    else:
        e2.append(example)
        
test = map(lambda x: x[0], e1)
test2 = map(lambda x: x[0], e2)
        
plt.scatter([inp[0] for inp in test], [inp[1] for inp in test], c='red')
plt.scatter([inp[0] for inp in test2], [inp[1] for inp in test2], c='blue')

plt.show()