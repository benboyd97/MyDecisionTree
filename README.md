# Introduction to ML - Decision Tree Coursework

This repository contains the implementation of the decision tree algorithm for the **Introduction to Machine Learning** course at Imperial College London. The project description can be found [here](assignment.pdf).

The work was done by Cornelius Braun, Ben Boyd, Shayaan and Ryan.

## How to use
1. `git clone git@gitlab.doc.ic.ac.uk:iml/decision-tree.git`
2. `main.py` to run the classifier (with and without pruning) on the noisy data set and a 10-fold cross-validation.


You can alternatively set the dataset and the number of folds for cross-validation using\
`main.py --data ["clean", "noisy"] --k [1,..,20]`

## Supplementary material
The project report can be found [here](link TBD). This report includes all main classifier metrics such as *precision*, *recall* and *F1*. Furthermore, we compare the performance of pruned and unpruned decision trees.