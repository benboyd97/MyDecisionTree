from DecisionTreeClassifier import DecisionTreeClassifier
from DataLoader import DataLoader
import numpy as np
from Evaluation import Evaluation
import argparse

parser = argparse.ArgumentParser(description="Decision tree classifier for the Introduction to Machine Learning Coursework 1")
parser.add_argument('--data', type=str, help="Data to classify (e.g. 'clean', 'noisy') ", default="noisy")
args = parser.parse_args()

print(f"Working on {args.data} data")

# Load the dataset
data_loader = DataLoader("wifi_db/" + args.data + "_dataset.txt") # TODO: we have to be careful here, as far as I know the forward slash does not work on Windows?
x, y = data_loader.load_data()
evaluation = Evaluation()

# TODO: maybe remove this since we have CV now?
repeat_count = 5
accuracy1 = np.zeros(repeat_count)
max_depth1 = np.zeros(repeat_count)
accuracy2 = np.zeros(repeat_count)
max_depth2 = np.zeros(repeat_count)
for i in range(repeat_count):
    # Split the dataset
    x_train, x_test, x_val, y_train, y_test, y_val = data_loader.split_dataset(x, y, 0.1, 0.1)

    # Train Decision Tree
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(x_train, y_train)

    # Calculate the accuracy of the Decision Tree
    accuracy1[i] = evaluation.evaluate(x_test, y_test, decision_tree)
    max_depth1[i] = decision_tree.depth

    # Prune Decision Tree
    decision_tree.prune(x_train, y_train, x_val, y_val)

    # Calculate the accuracy of the pruned Decision Tree
    accuracy2[i] = evaluation.evaluate(x_test, y_test, decision_tree)
    max_depth2[i] = decision_tree.depth

accuracy1 = sum(accuracy1) / repeat_count * 100
accuracy1 = "{:.2f}".format(accuracy1)
print(f"Before Pruning: Average accuracy of {accuracy1}% over {repeat_count} runs")
print('Average max depth', np.average(max_depth1))

accuracy2 = sum(accuracy2) / repeat_count * 100
accuracy2 = "{:.2f}".format(accuracy2)
print(f"After Pruning: Average accuracy of {accuracy2}% over {repeat_count} runs")
print('Average max depth', np.average(max_depth2))

decision_tree = DecisionTreeClassifier()
evaluation = Evaluation()

# make sure we just print two decimals
np.set_printoptions(precision=2)

print('\n \n 10 FOLD CROSS VALIDATION METRICS BEFORE PRUNING:')
confusion_matrix = evaluation.unnested_cross_validation(decision_tree, x, y, 10)
print('Confusion Matrix:')
print(confusion_matrix)
print(f'Accuracy: {evaluation.accuracy_from_confusion():.2f}')
print('Precision: ', evaluation.precision())
print('Recall: ', evaluation.recall())
print('F1: ', evaluation.F1())

decision_tree = DecisionTreeClassifier()

print('\n\n 10 FOLD CROSS VALIDATION METRICS AFTER PRUNING:')
confusion_matrix = evaluation.nested_cross_validation(decision_tree, x, y, 10)
print('Confusion Matrix:')
print(confusion_matrix)
print(f'Accuracy: {evaluation.accuracy_from_confusion():.2f}')
print('Precision: ', evaluation.precision())
print('Recall: ', evaluation.recall())
print('F1: ', evaluation.F1())
