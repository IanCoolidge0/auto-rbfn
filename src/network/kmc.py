import numpy as np
import random

def distance(x, y):
    return sum((xi - yi)**2 for (xi,yi) in zip(x,y))

def list_average(data):
    return map(lambda x: sum(x) / len(x), zip(*data))

#Perform k-means clustering on training data.
#Parameters:
#   training_data: list of input-output tuples representing all data
#   k: number of means
#   stages: number of stages of k-means clustering to run.
#Returns:
#   List of all centroids represented as arrays
#TODO: replace stages with a more precise algorithm
def k_means(training_data, k, stages):
    #Generate list of centroids randomly chosen from training data
    centroids = []
    beta = []
    for d in range(k):
        index = random.randint(0, len(training_data)-1)
        centroids.append(training_data[index])

    #Iterate algorithm once for each stage, not testing convergence
    for d in range(stages):
        clusters = [[] for e in range(k)]

        #determine minimal distance of training example and assign a cluster
        for i in range(len(training_data)):
            min_value = distance(centroids[0], training_data[i])
            min_index = 0
            for j in range(1, k):
                value = distance(centroids[j], training_data[i])
                if value < min_value:
                    min_value = value
                    min_index = j

            #generate clusters based on distance
            clusters[min_index].append(training_data[i])

        #set centroids to means of each cluster
        for i in range(k):
            if len(clusters[i]) == 0:
                centroids[i] = training_data[random.randint(0,len(training_data)-1)]
            else:
                centroids[i] = list_average(clusters[i])
                
    for i, centroid in enumerate(centroids):
        """ sigma = 0
        for point in clusters[i]:
            sigma += np.linalg.norm(np.array(point) - np.array(centroid)) / len(centroids)
        beta.append(1.0 / (2 * sigma * sigma)) """
        beta.append(0.3)
        
    return map(np.array, centroids), beta

            
            
