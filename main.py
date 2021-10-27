from DecisionTreeClassifier import DecisionTreeClassifier
from DataLoader import DataLoader
import numpy as np
from Evaluation import Evaluation

# Load the dataset
data_loader = DataLoader("wifi_db/noisy_dataset.txt")
x, y = data_loader.load_data()
evaluation=Evaluation()

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
    accuracy1[i]=evaluation.evaluate(x_test,y_test,decision_tree)
    max_depth1[i]=decision_tree.depth


    # Prune Decision Tree
    decision_tree.prune(x_train, y_train, x_val, y_val)

    # Calculate the accuracy of the pruned Decision Tree
    accuracy2[i]=evaluation.evaluate(x_test,y_test,decision_tree)
    max_depth2[i]=decision_tree.depth

accuracy1 = sum(accuracy1)/repeat_count * 100
accuracy1 = "{:.2f}".format(accuracy1)
print(f"Before Pruning: Average accuracy of {accuracy1}% over {repeat_count} runs")
print('Average max depth',np.average(max_depth1))

accuracy2 = sum(accuracy2)/repeat_count * 100
accuracy2 = "{:.2f}".format(accuracy2)
print(f"After Pruning: Average accuracy of {accuracy2}% over {repeat_count} runs")
print('Average max depth',np.average(max_depth2))



decision_tree = DecisionTreeClassifier()
evaluation=Evaluation()

print('\n \n 10 FOLD CROSS VALIDATION METRICS BEFORE PRUNING:')
confusion_matrix=evaluation.cross_validation(decision_tree,x,y,10)
print('Confusion Matrix:')
print(confusion_matrix)
print('Accuracy: ', evaluation.accuracy_from_confusion())
print('Precision: ',evaluation.precision())
print('Recall: ',evaluation.recall())
print('F1: ',evaluation.F1())

decision_tree = DecisionTreeClassifier()

print('\n\n 10 FOLD CROSS VALIDATION METRICS AFTER PRUNING:')
confusion_matrix=evaluation.pruned_cross_validation(decision_tree,x,y,10)
print('Confusion Matrix:')
print(confusion_matrix)
print('Accuracy: ', evaluation.accuracy_from_confusion())
print('Precision: ',evaluation.precision())
print('Recall: ',evaluation.recall())
print('F1: ',evaluation.F1())

