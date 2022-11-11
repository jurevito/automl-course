import numpy as np
import scipy as scp
import pandas as pd
import seaborn as sn
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

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
    'KNN'                : KNeighborsClassifier(),
    'Linear SVM'         : SVC(kernel="linear", C=0.025),
    'RBF SVM'            : SVC(gamma=2, C=1),
    'Gaussian Process'   : GaussianProcessClassifier(1.0 * RBF(1.0)),
    'Decision Tree'      : DecisionTreeClassifier(max_depth=5),
    'Random Forest'      : RandomForestClassifier(max_depth=5),
    'Neural Network'     : MLPClassifier(alpha=1, max_iter=1000),
    'AdaBoost'           : AdaBoostClassifier(),
    'Naive Bayes'        : GaussianNB(),
    'QDA'                : QuadraticDiscriminantAnalysis(),
    'Logistic Regression': LogisticRegression(max_iter=500),
}

category_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

def preprocess(df: pd.DataFrame):

    le = preprocessing.LabelEncoder()
    for column in category_columns:
        df[column] = le.fit_transform(df[column])

    return df

def find_baseline(df: pd.DataFrame) -> dict[str, float]:

    X = df.drop('HeartDisease', axis=1).values
    y = df['HeartDisease'].values

    scaler = StandardScaler()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name,  classifier in classifiers.items():
        scores = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            scaler.fit_transform(X_train)
            scaler.transform(X_test)

            model = classifier.fit(X=X_train, y=y_train)
            y_pred = model.predict(X_test)

            score = accuracy_score(y_test, y_pred)
            scores.append(score)

        results[name] = sum(scores) / len(scores)

    return results


if __name__ == '__main__':
    df = pd.read_csv("./heart_failure.csv")
    df = preprocess(df)
    find_baseline(df)
