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
                if sample[node["attribute"]] < node["value"]:
                    node = node["left"]
                else:
                    node = node["right"]
            y.append(node["value"])
        
        return y

    def calculate_entropy(self, y):

        counts = [0]*4
        total = len(y)

        for sample in y:
            counts[int(sample)-1] += 1

        entropy = 0
        for label in range(4):
            if counts[label] != 0:
                p_label = counts[label]/total
                entropy -= (p_label * math.log2(p_label))

        return entropy

    def find_split(self, x, y):
        # choose the attribute and value that results in the highest information gain

        total_entropy = self.calculate_entropy(y)
        data_samples = np.c_[x, y]
        
        best_gain = 0
        best_split_val = None
        best_split_attrib = None
        l_data_x = None
        l_data_y = None
        r_data_x = None
        r_data_y = None
        
        for attrib in range(x.shape[1]):

            # sort data according to attribute in question
            sorted_attrib = data_samples[np.argsort(data_samples[:,attrib])]

            for sample in range(1, len(sorted_attrib)):
                if sorted_attrib[sample][-1] !=  sorted_attrib[sample-1][-1]:
                    split_value = (sorted_attrib[sample][attrib] + sorted_attrib[sample-1][attrib]) / 2
                    
                    # split on attrib with val split_val
                    remainder_l = ((sample)/len(sorted_attrib)) * self.calculate_entropy(sorted_attrib[:sample][:, -1])
                    remainder_r = ((len(sorted_attrib)-sample)/len(sorted_attrib)) * self.calculate_entropy(sorted_attrib[sample:][:, -1])
                    gain = total_entropy - (remainder_l + remainder_r)

                    if gain > best_gain:
                        best_gain = gain
                        best_split_val = split_value
                        best_split_attrib = attrib
                        l_data_x = sorted_attrib[:sample][:, :-1]
                        l_data_y = sorted_attrib[:sample][:, -1]
                        r_data_x = sorted_attrib[sample:][:, :-1]
                        r_data_y = sorted_attrib[sample:][:, -1]

        return best_split_val, best_split_attrib, l_data_x, l_data_y, r_data_x, r_data_y


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