import numpy as np
import scipy as scp
import pandas as pd
import seaborn as sn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

classifiers = {
    'KNN': KNeighborsClassifier(3),
    'Linear SVM': SVC(kernel="linear", C=0.025),
    'RBF SVM': SVC(gamma=2, C=1),
    'Gaussian Process': GaussianProcessClassifier(1.0 * RBF(1.0)),
    'Decision Tree': DecisionTreeClassifier(max_depth=5),
    'Random Forest': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    'Neural Network': MLPClassifier(alpha=1, max_iter=1000),
    'AdaBoost': AdaBoostClassifier(),
    'Naive Bayes': GaussianNB(),
    'QDA': QuadraticDiscriminantAnalysis(),
    'Logistic Regression': LogisticRegression(),
}

def find_baseline(df):
    pass

if __name__ == '__main__':
    pass