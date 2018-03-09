import numpy as np

def load_csv(path, input_size, output_size):
    training_data = []
    training_outputs = []
    
    with open(path) as f:
        content = f.readlines()

    for line in content:
        tokens = line.split(",")
        
        training_data.append(map(float, tokens[:input_size]))
        training_outputs.append(map(float, tokens[input_size:]))

    return zip(training_data, training_outputs)

  