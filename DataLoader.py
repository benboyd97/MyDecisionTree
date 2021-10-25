import numpy as np
from numpy.random import default_rng

class DataLoader:
    def __init__(self, path):
        self.path = path
    
    def load_data(self):
        dataset = np.loadtxt(self.path)
        x = dataset[:, :-1]
        y = dataset[:, -1]

        return x, y

    def split_dataset(self, x, y, test_proportion, val_proportion, random_generator=default_rng()):      
        shuffled_indicies = random_generator.permutation(len(x))
        n_test = round(len(x) * test_proportion)
        n_val = round(len(x) * val_proportion)
        n_train = len(x) - n_test - n_val

        x_train = x[shuffled_indicies[:n_train]]
        y_train = y[shuffled_indicies[:n_train]]

        x_test = x[shuffled_indicies[n_train:n_train+n_test]]
        y_test = y[shuffled_indicies[n_train:n_train+n_test]]

        x_val = x[shuffled_indicies[n_train+n_test:]]
        y_val = y[shuffled_indicies[n_train+n_test:]]

        return x_train, x_test, x_val, y_train, y_test, y_val