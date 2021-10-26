import numpy as np
from numpy.random import default_rng

class Evaluation:

    def __init__(self):
        self.matrix=[[0]]


    def evaluate(self,x_test,y_test,tree):


        #predict rooms based on test signals
        y_preds=tree.predict(x_test)

        #return accuracy
        return np.count_nonzero(y_preds == y_test) / y_test.shape[0]



    def validation_error(self, y_preds, y_test):
        return np.count_nonzero(y_preds != y_test) / y_test.shape[0]




    def cross_validation(self,tree,x,y,k):

        rows,columns=np.shape(x)
        split_ids=self.k_split(k,rows)

        folds=self.find_folds(k,split_ids)



        y_actual,y_pred=self.train_test_folds(tree,x,y,k,folds)


        
        self.matrix=self.confusion_matrix(y_actual,y_pred)


        return self.matrix


    def k_split(self,k, rows, random_generator=default_rng()):


        # generate rows-1 ids and shuffle them
        shuffled_ids = random_generator.permutation(rows)

        # split the shuffled ids into k equal groups
        split_ids = np.array_split(shuffled_ids, k)


        return split_ids


    def find_folds(self,k,split_ids):

        folds=[]

        for i in range(k):

            test_ids = split_ids[i]

            train_ids = np.hstack(split_ids[:i] + split_ids[i+1:])

            folds.append([train_ids, test_ids])

        
        return folds


    def train_test_folds(self,tree,x,y,k,folds):

        y_pred=np.array([])
        y_actual=np.array([])


        for i, (train_ids, test_ids) in enumerate(folds):
            # get the dataset from the correct splits
            x_train = x[train_ids, :]
            y_train = y[train_ids]
            x_test = x[test_ids, :]
            y_test = y[test_ids]

            
            tree.fit(x_train,y_train)

            y_pred=np.append(y_pred,tree.predict(x_test))

            y_actual=np.append(y_actual,y_test)


        return y_actual,y_pred


    def confusion_matrix(self,y_gold, y_prediction, class_labels=[1,2,3,4]):



        confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)

        for (i, label) in enumerate(class_labels):
        # get predictions where the ground truth is the current class label
            indices = (y_gold == label)
            gold = y_gold[indices]
            predictions = y_prediction[indices]

        # quick way to get the counts per label
            (unique_labels, counts) = np.unique(predictions, return_counts=True)

        # convert the counts to a dictionary
            frequency_dict = dict(zip(unique_labels, counts))

        # fill up the confusion matrix for the current row
            for (j, class_label) in enumerate(class_labels):
                confusion[i, j] = frequency_dict.get(class_label, 0)


        

        return confusion/len(y_gold)

    
    def accuracy_from_confusion(self):
        confusion=self.matrix
        if np.sum(confusion) > 0:

            return np.sum(np.diag(confusion)) / np.sum(confusion)
        else:
            return 0.



    def TPTN_FPFN(self,class_val):
        matrix=self.matrix
        
        TP=matrix[class_val,class_val]
        TN=np.sum(np.diag(matrix)) - TP

        FP=sum(matrix[:,class_val])-TP
        FN=sum(matrix[class_val,:])-TP

    
        return TP,TN,FP,FN

    
    def recall(self):
        
        rows,columns=np.shape(self.matrix)

        recall_array=np.zeros(rows)

        for i in range(rows):

            TP,TN,FP,FN=self.TPTN_FPFN(i)


            recall_array[i]= TP/(TP+FN)


        return recall_array


    def precision(self):
        
        rows,columns=np.shape(self.matrix)

        precision_array=np.zeros(rows)

        for i in range(rows):

            TP,TN,FP,FN=self.TPTN_FPFN(i)


            precision_array[i]= TP/(TP+FP)


        return precision_array








            

            


    # Train the KNN (we'll use one nearest neighbour)
            














    