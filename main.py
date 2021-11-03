from DecisionTreeClassifier import DecisionTreeClassifier
from DataLoader import DataLoader
import graph
import numpy as np
from Evaluation import Evaluation
import argparse
import copy

parser = argparse.ArgumentParser(description="Decision tree classifier for the Introduction to Machine Learning Coursework 1")
parser.add_argument('--data', type=str, help="Data to classify (e.g. 'clean', 'noisy') ", default="noisy")
parser.add_argument('--k', type=int, help="k for cross-validation (e.g. 10) ", default=10, choices=range(1, 100))
args = parser.parse_args()

print(f"Working on {args.data} data, using {args.k} folds")

# Load the dataset
data_loader = DataLoader("wifi_db/" + args.data + "_dataset.txt") # TODO: we have to be careful here, as far as I know the forward slash does not work on Windows?
x, y = data_loader.load_data()
evaluation = Evaluation()

base_accuracy = base_depth = prune_accuracy = prune_depth = 0

# Split the dataset
x_train, x_test, x_val, y_train, y_test, y_val = data_loader.split_dataset(x, y, 0.1, 0.1)

# Train Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)

# Prune Decision Tree
# TODO: Maybe change the prune function to return another decision tree so we dont have to copy the base tree to be able to graph it
pruned_tree = copy.deepcopy(decision_tree)
pruned_tree.prune(x_train, y_train, x_val, y_val)

graph.plot(decision_tree, pruned_tree)

decision_tree = DecisionTreeClassifier()
evaluation = Evaluation()

# make sure we just print two decimals
np.set_printoptions(precision=2)

print(f'\n \n {args.k} FOLD CROSS VALIDATION METRICS BEFORE PRUNING:')
confusion_matrix = evaluation.cross_validation(decision_tree, x, y, args.k)
print('Confusion Matrix:')
print(confusion_matrix)
print(f'Accuracy: {evaluation.accuracy_from_confusion():.2f}')
print('Precision: ', evaluation.precision())
print('Recall: ', evaluation.recall())
print('F1: ', evaluation.F1())
print(f'Average Max Depth: {evaluation.average_max_depth():.2f}')

decision_tree = DecisionTreeClassifier()
evaluation = Evaluation()

print(f'\n\n {args.k} FOLD CROSS VALIDATION METRICS AFTER PRUNING:')
confusion_matrix = evaluation.nested_cross_validation(decision_tree, x, y, args.k)
print('Confusion Matrix:')
print(confusion_matrix)
print(f'Accuracy: {evaluation.accuracy_from_confusion():.2f}')
print('Precision: ', evaluation.precision())
print('Recall: ', evaluation.recall())
print('F1: ', evaluation.F1())
print(f'Average Max Depth: {evaluation.average_max_depth():.2f}')
