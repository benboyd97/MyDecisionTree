import numpy as np

class DecisionTreeClassifier:
    def __init__(self):
        self.root = None
        self.depth = 0
    
    # Create the decision tree
    def fit(self, x, y):
        self.root, self.depth = self.decision_tree_learning(x, y, 0)

    # Make predictions for the sample by traversing the decision tree
    def predict(self, x):
        predictions = []
        for sample in x:
            node = self.root
            while node["left"] and node["right"]:
                node = node["left"] if sample[node["attribute"]] <= node["value"] else node["right"]
            predictions.append(node["value"])
        return np.array(predictions)

    # Calculate the entropy for a given dataset
    def get_entropy(self, data: np.array):
        _, label_counts = np.unique(data[:,-1], return_counts=True)
        total_count = data.shape[0]
        H = -np.sum((label_counts/total_count)*np.log2(label_counts/total_count))
        return H

    # Find the optimal attribute and value for splitting the data
    def find_split(self, data: np.array):
        split_node = {"attribute":-1, "value":-1, "left":np.zeros(1), "right":np.zeros(1), "entropy_gain":-1}
        
        H_all = self.get_entropy(data)
        # Iterate over each attribute
        for i, attribute in enumerate(data[:,:-1].T):
            label_values = np.unique(np.sort(attribute))                # sort the labels and duplicates
            splits = label_values[:-1] + (np.diff(label_values) / 2)    # create split points from distances between the values
            
            # Test each split value
            for split in splits:
                # Segment the data according to the split
                left = data[np.where(data[:,i] <= split)]
                right = data[np.where(data[:,i] > split)]

                # Calculate the entropy gain
                remainder_left = left.shape[0]*self.get_entropy(left)
                remainder_right = right.shape[0]*self.get_entropy(right)
                gain = H_all - ((remainder_left + remainder_right) / attribute.shape[0])
                
                # If the entropy gain is higher then use this split
                if gain > split_node["entropy_gain"]:
                    split_node["attribute"] = i
                    split_node["value"] = split
                    split_node["left"] = left
                    split_node["right"] = right
                    split_node["entropy_gain"] = gain
        return split_node

    def decision_tree_learning(self, x:np.array, y:np.array, depth:int):
        # If all samples in the set belong to the same class then we are done segmenting this set
        if len(np.unique(y)) == 1:
            leaf_node = {"attribute": None, "value": y[0], "left":None, "right":None, "leaf":True}
            return leaf_node, depth
            
        # Split the set according to the attribute the gives the greatest information gain
        data = np.c_[x,y]
        split = self.find_split(data)
        l_branch, l_depth = self.decision_tree_learning(split["left"][:,:-1], split["left"][:,-1], depth+1)
        r_branch, r_depth = self.decision_tree_learning(split["right"][:,:-1], split["right"][:,-1], depth+1)        
        node = {"attribute":split["attribute"], "value":split["value"], "left":l_branch, "right":r_branch, "leaf":False}
        return node, max(l_depth, r_depth)