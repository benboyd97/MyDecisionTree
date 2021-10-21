from DecisionTreeClassifier import DecisionTreeClassifier
from DataLoader import DataLoader
import numpy as np

# Load and split dataset
data_loader = DataLoader("wifi_db/clean_dataset.txt")
x, y = data_loader.load_data()
x_train, x_test, y_train, y_test = data_loader.split_dataset(x, y, 0.2)

# Train Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)

# Test Decision Tree
y_preds = decision_tree.predict(x_test)

# Calculate the accuracy of the Decision Tree
print(f"{np.count_nonzero(y_preds == y_test) / y_preds.shape[0] * 100}%")