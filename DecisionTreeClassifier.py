import numpy as np
import math

class DecisionTreeClassifier:
    def __init__(self):
        self.root = None
        self.depth = 0
    
    def fit(self, x, y):
        self.root, self.depth = self.decision_tree_learning(x, y, 0)

    def predict(self, x):
        
        y = []

        for sample in x:
            node = self.root
            while node["left"] and node["right"]:
                if sample[node["attribute"]] <= node["value"]:
                    node = node["left"]
                else:
                    node = node["right"]
            y.append(node["value"])
        
        return y

    def get_entropy(self, data: np.array):
        label_values, label_counts = np.unique(data[:,-1], return_counts=True)
        H = -np.sum((label_counts/data.shape[0])*np.log2(label_counts/data.shape[0]))
        return H

    def find_split(self, data: np.array):
        H_all = self.get_entropy(data)
        #print("H", H_all)
        split_node = {"attribute":-1, "value":-1, "left":np.zeros(1), "right":np.zeros(1), "entropy_gain":-1}
        
        # columns represent features
        for i, attribute in enumerate(data[:,:-1].T):
            # TODO: eliminate the split points for equal neighbours
            splits = np.sort(attribute)[:-1] + (np.diff(np.sort(attribute)) / 2)
            
            # go over each possible split value
            for split in splits:
                H_gain = H_all
                left = data[np.where(data[:,i] <= split)]
                right = data[np.where(data[:,i] > split)]
                H_gain += -(left.shape[0]*self.get_entropy(left)+right.shape[0]*self.get_entropy(right)) / attribute.shape[0]
                
                # check for information gain
                if H_gain > split_node["entropy_gain"]:
                    split_node["attribute"] = i
                    split_node["value"] = split
                    split_node["left"] = left
                    split_node["right"] = right
                    split_node["entropy_gain"] = H_gain
                    
        return split_node



    def decision_tree_learning(self, x, y, depth):
        if all(i==y[0] for i in y):
            return {"attribute": None, "value": y[0], "left": None, "right": None}, depth
        else:
            split_val, split_attrib, l_data_x, l_data_y, r_data_x, r_data_y = self.find_split(x, y)
            
            node = {"attribute": split_attrib, "value": split_val, "left": None, "right": None}
            l_node, l_depth = self.decision_tree_learning(l_data_x, l_data_y, depth+1)
            node["left"] = l_node
            r_node, r_depth = self.decision_tree_learning(r_data_x, r_data_y, depth+1)
            node["right"] = r_node

            return node, max(l_depth, r_depth)

    def decision_tree_learning(self, x: np.array, y:np.array, depth: int):
        if len(np.unique(y)) == 1:
            leaf_node = {"attribute": None, "value": y[0], "left":None, "right":None, "leaf":True}
            return leaf_node, depth
        else:
            data = np.c_[x,y]
            split = self.find_split(data)
            l_branch, l_depth = self.decision_tree_learning(split["left"][:,:-1], split["left"][:,-1], depth+1)
            r_branch, r_depth = self.decision_tree_learning(split["right"][:,:-1], split["right"][:,-1], depth+1)        
            node = {"attribute":split["attribute"], "value":split["value"], "left":l_branch, "right":r_branch, "leaf":False}

            return node, max(l_depth, r_depth)