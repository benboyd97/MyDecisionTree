from DecisionTreeClassifier import DecisionTreeClassifier
from DataLoader import DataLoader

# Load and split dataset
data_loader = DataLoader("wifi_db/clean_dataset.txt")
x, y = data_loader.load_data()
x_train, x_test, y_train, y_test = data_loader.split_dataset(x, y, 0.2)

# Train Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)

# Test Decision Tree
y_preds = decision_tree.predict(x_test)

correct = 0
for i in range(len(y_preds)):
    if y_preds[i] == y_test[i]:
        correct += 1

print(correct/len(y_preds))