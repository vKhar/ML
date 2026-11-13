import matplotlib.pyplot as plt
import numpy as np

'''
data input code
'''

def read_labelled_data(filename):
    """Reads csv and returns tuple of float features and str labels (last column)."""
    features = []
    labels = []
    with open(filename) as f:
        lines = f.read().split('\n')
        for line in lines:
            pieces = line.split(',')
            features.append(list(map(float, pieces[:-1])))
            labels.append(pieces[-1])
    return features, labels

def np_read_labelled_data(filename:str)->tuple[np.ndarray, np.ndarray]:
    return (
        np.loadtxt("labelled.csv",delimiter=",",usecols=(0,1)),
        np.loadtxt("labelled.csv",delimiter=",",dtype=str,usecols=(-1,)) 
    )


''' 
plot labeled data points 
'''

import matplotlib.pyplot as plt
def plot_labelled_data(features, labels, y_units="bpm", x_units="seconds", y_label="song tempo", x_label="song length"):
    """Plot labelled data using different shapes and colours for each label."""
    # Note: You do not need to understand how the following matplotlib code works.
    shapes = ['o', 's', '^', 'D', 'x']
    colours = ['red', 'blue', 'green', 'orange', 'purple']

    # Use hard-coded size values for this example.
    plt.figure(figsize=(6, 4.5))
    
    label_names = sorted(set(labels))
    for i in range(len(label_names)):
        points = [features[j] for j in range(len(features)) if labels[j] == label_names[i]]
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        plt.scatter(x, y, c=colours[i % len(colours)], 
                    marker=shapes[i % len(shapes)], s=120, label=label_names[i])
    plt.xlabel(f'{x_label} ({x_units})')
    plt.ylabel(f'{y_label} ({y_units})')
    plt.legend()
    plt.grid(True)
    plt.show()

"""
Plot unlabelled data point
"""
def plot_data(data, y_units="bpm", x_units="seconds"):

    # Note: You do not need to understand how the following matplotlib code works.
    # Use hard-coded size values for this example.
    plt.figure(figsize=(6, 4.5))
    
    x = [p[0] for p in data]
    y = [p[1] for p in data]
    plt.scatter(x, y, s=120)
    plt.xlabel(f'song length ({x_units})')
    plt.ylabel(f'song tempo ({y_units})')
    plt.grid(True)
    plt.show()
'''
plot training and test data points side by side
'''
def plot_train_test(train_features, test_features, 
                    train_labels, test_labels, 
                    labels):
    shapes = ['o', 's', '^', 'D', 'x']
    colours = ['red', 'blue', 'green', 'orange', 'purple']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    for i,label in enumerate(np.unique(labels)):
        mask = train_labels == label
        axes[0].scatter(train_features[mask, 0], train_features[mask, 1], label=label, 
                        c=colours[i % len(colours)], 
                        marker=shapes[i % len(shapes)])
    axes[0].set_title("Training Data")
    axes[0].set_xlabel("Song Length (seconds)")
    axes[0].set_ylabel("Tempo (bpm)")
    axes[0].legend()

    for i,label in enumerate(np.unique(labels)):
        mask = test_labels == label
        axes[1].scatter(test_features[mask, 0], test_features[mask, 1], label=label,
                        c=colours[i % len(colours)], 
                        marker=shapes[i % len(shapes)])
    axes[1].set_title("Test Data")
    axes[1].set_xlabel("Song Length (seconds)")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def plot_train_test_wrapper(
    training_features, 
    training_labels, 
    test_features, 
    test_labels,
    labels
):
        
    plot_train_test(
        np.array(training_features), 
        np.array(test_features),
        np.array(training_labels),
        np.array(test_labels),
        np.array(labels)
    )
        
def plot_folds(fold_indices, features, labels):
    features = np.array(features)
    labels = np.array(labels)
    for f in range(len(fold_indices)):
            test_indices = fold_indices[f]
            # train_indices = [ i for i in range(len(features)) if i not in test_indices]
            train_indices = np.arange(len(features))[ ~np.isin(np.arange(len(features)),test_indices )]
            print(f"fold {f+1}")
            plot_train_test( features[ train_indices], features[test_indices], 
                     labels [ train_indices], labels[test_indices], labels)


## knn code (non-numpy) code

def distance_squared(point1, point2):
    """
    point1, point2 are tuples (x,y)
    Returns distance squared between two given 2D points.
    """
    return (point2[1] - point1[1])**2 + (point2[0] - point1[0])**2

def predict(k, training_features, training_labels, test_point):
    '''
    training_features : [[float,float],..]
    training_labels :[ str, str, ..]
    Returns the most common label of the k nearest training points to test_point.
    '''
    
    # Calculate distances from test_point to every point in training data.
    distances = []
    for i in range(len(training_features)):
        dist = distance_squared(training_features[i], test_point)
        distances.append((dist, training_labels[i])) # later sort by dist
    
    # Get labels of the k nearest training points.
    distances.sort() 
    k_nearest_labels = [d[1] for d in distances[:k]]
    # print(k_nearest_labels)
    # print( max(k_nearest_labels, key = k_nearest_labels.count) )
    
    # Return the most common label.
    knn_count = { k: k_nearest_labels.count(k) for k in set (k_nearest_labels)}
    return max ( knn_count, key=knn_count.get) # max of the value in the key,value

    # equivalent code, for each unique labels (set), count that label in the list, get the max of those count
    return max(set(k_nearest_labels), key=k_nearest_labels.count)

def predict_probabilities(k, training_features, 
                          training_labels, test_point):
    """Returns a tuple of the most common label of the k
       nearest training points to test_point and a dict
       mapping labels to their estimated probability."""
    
    
    # Calculate distances from test_point to every point in training data.
    distances = []
    for i in range(len(training_features)):
        dist = distance_squared(training_features[i], test_point)
        distances.append((dist, training_labels[i])) # later sort by dist
    
    # Get labels of the k nearest training points.
    distances.sort() 
    k_nearest_labels = [d[1] for d in distances[:k]]
    
    # Return the most common label.
    keys = set (k_nearest_labels)
    knn_count = { k: k_nearest_labels.count(k) for k in keys} 

    knn_total = sum(knn_count.values()) # same as k
    probabilities = {l: knn_count.get(l,0)/knn_total for l in set(training_labels)}
    # :
    #     count = knn_count.get(l,0)
    #     print(f"P({l})={count}/{knn_total}")
    return max ( knn_count, key=knn_count.get), probabilities
   

'''
print numpy array and meta-data
'''
def array_print(array:np.ndarray):
    print(f"{array}")
    print(f"{'dtype':<8}{'shape':<10}{'dim':<5}{'size':<10}")
    print(f"{str(array.dtype):<8}{str(array.shape):<10}{array.ndim:<5}{array.size}")
    print(f"-"*28)