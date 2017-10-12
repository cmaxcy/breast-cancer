"""
Data used belongs to the Wisconsin Breast Cancer Dataset
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

Supervised learning is performed on data, with the goal being to classify cases
as malignant or benign.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# TODO:
# - Consider imputing missing data
# - Consider using grid search to obtain SVM hyperparameters
# - Consider using ensemble method after (or before?) grid search

# Prepare data. Column with missing data is not used
# 0 will represent benign, 1 malignant
data_file = "breast-cancer-wisconsin.data"
raw_data = pd.read_csv(data_file)
del raw_data["id"]
del raw_data["bare-nuclei"]
Y = np.array(raw_data["class"])
Y = Y / 2 - 1
del raw_data["class"]
X = np.array(raw_data)

# Split data randomly into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Fit Support Vector Machine to training data and report accuracy on testing data
clas = SVC()
clas.fit(x_train, y_train)
print(clas.score(x_test, y_test))
