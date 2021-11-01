import numpy as np
from numpy.random import default_rng

class Evaluation:
    def __init__(self):
        self.matrix = [[0]]

    def evaluate(self, x_test, y_test, tree):
        # predict rooms based on test signals
        y_preds = tree.predict(x_test)

        # return accuracy
        return np.count_nonzero(y_preds == y_test) / y_test.shape[0]

        #calculate error based on the number of wrongly classified rooms
    def validation_error(self, y_preds, y_test):
        return np.count_nonzero(y_preds != y_test) / y_test.shape[0]

        #trains the model on k different folds and finds average confusion matrix
    def cross_validation(self, tree, x, y, k):
        rows, columns = np.shape(x)
        split_ids = self.k_split(k, rows)
        folds = self.find_folds(k, split_ids)
        confusion = [[0]]
        depth=0
        for i, (train_ids, test_ids) in enumerate(folds):
            x_train = x[train_ids, :]
            y_train = y[train_ids]
            x_test = x[test_ids, :]
            y_test = y[test_ids]
            tree.fit(x_train, y_train)
            y_pred = tree.predict(x_test)
            depth+=tree.calc_depth(tree.root)
            confusion += self.confusion_matrix(y_test,y_pred)
        self.average_depth=depth/k
        self.matrix=confusion/k
        return self.matrix

    #option 2 nested cross validation finds the average confusion matrix based on k outer folds and k-1 inner folds
    def nested_cross_validation(self, tree, x, y, k):
        rows, columns = np.shape(x)
        split_ids = self.k_split(k, rows)
        outer_confusion = [[0]]
        outer_depth=0
        for i in range(k):
            inner_confusion = [[0]]
            inner_depth=0
            test_ids = split_ids[i]
            x_test= x[test_ids, :]
            y_test = y[test_ids]
            new_split_ids = split_ids[:i] + split_ids[i + 1:]
            folds = self.find_folds(k-1, new_split_ids)
            for j, (train_ids, val_ids) in enumerate(folds):
                x_train = x[train_ids, :]
                y_train = y[train_ids]
                x_val = x[val_ids, :]
                y_val = y[val_ids]
                tree.fit(x_train, y_train)
                tree.prune(x_train, y_train, x_val, y_val)
                y_pred = tree.predict(x_test)
                inner_depth+=tree.calc_depth(tree.root)
                inner_confusion += self.confusion_matrix(y_test, y_pred)
            outer_confusion+= inner_confusion/(k-1)
            outer_depth+=inner_depth/(k-1)
        self.average_depth=outer_depth/k
        self.matrix=outer_confusion/k
        return self.matrix

    #split data ids into k evenly sized chunks 
    def k_split(self, k, rows, random_generator=default_rng()):
        shuffled_ids = random_generator.permutation(rows)
        split_ids = np.array_split(shuffled_ids, k)
        return split_ids

    #stores every permutation of having one chunk test set and k-1 chunks of training set
    def find_folds(self, k, split_ids):
        folds = []
        for i in range(k):
            test_ids = split_ids[i]
            train_ids = np.hstack(split_ids[:i] + split_ids[i + 1:])
            folds.append([train_ids, test_ids])

        return folds


    #finds the confusion matrix
    def confusion_matrix(self, y_actual, y_pred, class_labels=None):
        if class_labels is None:
            class_labels = np.unique(np.concatenate((y_pred, y_actual)))

        confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)

        for label in class_labels:
            if label in y_actual:
                for pred in y_pred[y_actual == label]:
                    confusion[int(label)-1, int(pred)-1] += 1

        return confusion

    #finds accuracy directly from confusion matrix    
    def accuracy_from_confusion(self):
        confusion = self.matrix
        if np.sum(confusion) > 0:
            return np.trace(confusion) / np.sum(confusion)
        else:
            return 0.


    #finds recall for each class directly from confusion matrix
    def recall(self):
        confusion = self.matrix
        return np.diagonal(confusion) / np.sum(confusion, axis=1)

    #finds precision for each class directly from confusion matrix
    def precision(self):
        confusion = self.matrix
        return np.diagonal(confusion) / np.sum(confusion, axis=0)

    #finds F1 for each class using precision and recall
    def F1(self):
        precision = self.precision()
        recall = self.recall()
        return (2 * precision * recall) / (precision + recall)


    #outputs average depth
    def average_max_depth(self):
        return self.average_depth
