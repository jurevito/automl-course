import numpy as np
import scipy as scp
import pandas as pd
import seaborn as sn
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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
    'KNN'                : KNeighborsClassifier(3),
    'Linear SVM'         : SVC(kernel="linear", C=0.025),
    'RBF SVM'            : SVC(gamma=2, C=1),
    'Gaussian Process'   : GaussianProcessClassifier(1.0 * RBF(1.0)),
    'Decision Tree'      : DecisionTreeClassifier(max_depth=5),
    'Random Forest'      : RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    'Neural Network'     : MLPClassifier(alpha=1, max_iter=1000),
    'AdaBoost'           : AdaBoostClassifier(),
    'Naive Bayes'        : GaussianNB(),
    'QDA'                : QuadraticDiscriminantAnalysis(),
    'Logistic Regression': LogisticRegression(),
}


def find_baseline(df):
    df.dropna(subset=['HeartDisease'], inplace=True)

    X = df.drop('HeartDisease', axis=1)
    y = df.HeartDisease

    le = preprocessing.LabelEncoder()

    for column in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
        X[column] = le.fit_transform(X[column])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2021)

    accuracies = {}
    print(X_train.head())

    for classifier in classifiers.values():
        model = classifier.fit(X=X_train, y=y_train)
        predictions = model.predict(X_test)
        accuracies[model] = accuracy_score(y_test, predictions)

    print(accuracies)


if __name__ == '__main__':
    find_baseline(pd.read_csv("./data/heart_failure.csv"))
