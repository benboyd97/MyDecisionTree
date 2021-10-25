from DecisionTreeClassifier import DecisionTreeClassifier
from DataLoader import DataLoader
import numpy as np

# Load the dataset
data_loader = DataLoader("wifi_db/noisy_dataset.txt")
x, y = data_loader.load_data()

repeat_count = 5
accuracy = 0
for i in range(repeat_count):

    # Split the dataset
    x_train, x_test, y_train, y_test = data_loader.split_dataset(x, y, 0.2)

    # Train Decision Tree
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(x_train, y_train)

    # Test Decision Tree
    y_preds = decision_tree.predict(x_test)

    # Calculate the accuracy of the Decision Tree
    accuracy += np.count_nonzero(y_preds == y_test) / y_preds.shape[0]

accuracy = accuracy/repeat_count * 100
accuracy = "{:.2f}".format(accuracy)
print(f"Average accuracy of {accuracy}% over {repeat_count} runs")