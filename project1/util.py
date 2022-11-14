import numpy as np
import scipy as scp
import pandas as pd
import seaborn as sn
from matplotlib import pyplot
from hyperopt import fmin, space_eval, hp, tpe, STATUS_OK, Trials
from typing import Protocol

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold

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

class ScikitModel(Protocol):
    def fit(self, X, y, sample_weight=None): ...
    def predict(self, X): ...
    def score(self, X, y, sample_weight=None): ...
    def set_params(self, **params): ...

models = {
    'KNN'                : KNeighborsClassifier(),
    'SVM'                : SVC(kernel="linear", C=0.025, random_state=42),
    'Gaussian Process'   : GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    'Decision Tree'      : DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest'      : RandomForestClassifier(max_depth=5, random_state=42),
    'Neural Network'     : MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    'AdaBoost'           : AdaBoostClassifier(random_state=42),
    'Naive Bayes'        : GaussianNB(),
    'QDA'                : QuadraticDiscriminantAnalysis(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
}

classifiers = {
    'KNN'                : KNeighborsClassifier,
    'SVM'                : SVC,
    'Gaussian Process'   : GaussianProcessClassifier,
    'Decision Tree'      : DecisionTreeClassifier,
    'Random Forest'      : RandomForestClassifier,
    'Neural Network'     : MLPClassifier,
    'AdaBoost'           : AdaBoostClassifier,
    'Naive Bayes'        : GaussianNB,
    'QDA'                : QuadraticDiscriminantAnalysis,
    'Logistic Regression': LogisticRegression,
}

search_spaces = {
    'Random Forest': {
        'n_estimators': hp.randint('n_estimators', 50, 150),
        'criterion': hp.choice('criterion', ['gini', 'entropy', 'log_loss']),
        'max_depth': hp.randint('max_depth', 10, 200),
        'min_samples_split': hp.randint('min_samples_split', 2, 3),
        'min_samples_leaf': hp.randint('min_samples_leaf', 1, 2),
        'min_weight_fraction_leaf': hp.uniform('min_weight_fraction_leaf', 0, 0.05),
        'max_features': hp.choice('max_features', ['sqrt', 'log2']),
    },

    'Logistic Regression': {
        'penalty': hp.choice('penalty', ['l2']),
        'C': hp.uniform('C', 0.90, 1.0),
        'solver': hp.choice('solver', ['newton-cg', 'lbfgs', 'liblinear']),
        'max_iter': hp.choice('max_iter', [1000])
    },

    'AdaBoost': {
        'n_estimators': hp.randint('n_estimators', 20, 250),
        'learning_rate': hp.uniform('learning_rate', 0, 2),
        'algorithm': hp.choice('algorithm', ['SAMME', 'SAMME.R'])
    },

    'Neural Network': {
        'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [(100,)]),
        'activation': hp.choice('activation', ['identity', 'logistic', 'tanh', 'relu']),
        'solver': hp.choice('solver', ['lbfgs', 'sgd', 'adam']),
        'alpha': hp.uniform('alpha', 0, 1e-3),
        'learning_rate': hp.choice('learning_rate', ['constant', 'invscaling', 'adaptive']),
        'max_iter': hp.choice('max_iter', [100])
    },
}

category_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

def preprocess(df: pd.DataFrame):

    le = LabelEncoder()
    for column in category_columns:
        df[column] = le.fit_transform(df[column])

    return df

def find_baseline(df: pd.DataFrame) -> dict[str, float]:

    X = df.drop('HeartDisease', axis=1).values
    y = df['HeartDisease'].values

    scaler = StandardScaler()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name,  model in models.items():
        scores = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            scaler.fit_transform(X_train)
            scaler.transform(X_test)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            score = accuracy_score(y_test, y_pred)
            scores.append(score)

        results[name] = sum(scores) / len(scores)

    return results

def create_objective(classifier_name: str, df: pd.DataFrame, scale_values: bool):

    X = df.drop('HeartDisease', axis=1).values
    y = df['HeartDisease'].values

    def objective(search_space):

        classifier = classifiers[classifier_name]

        try:
            model: ScikitModel = classifier(**search_space, random_state=42)
        except:
            model: ScikitModel = classifier(**search_space)

        scaler = StandardScaler()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if scale_values:
                scaler.fit_transform(X_train)
                scaler.transform(X_test)

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            scores.append(score)

        avg_score = sum(scores) / len(scores)

        return {'loss': 1 - avg_score, 'status': STATUS_OK}

    return objective

def optimize_hyperparams(classifier_name: str, df: pd.DataFrame, max_evals: int, scale_values: bool = True):

    trials = Trials()

    optimized_params = fmin(
        fn=create_objective(classifier_name, df, scale_values=scale_values),
        space=search_spaces[classifier_name],
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials
    )

    losses = [ trial['result']['loss'] for trial in trials ]

    return space_eval(search_spaces[classifier_name], optimized_params), losses

if __name__ == '__main__':
    df = pd.read_csv("./heart_failure.csv")
    df = preprocess(df)
    find_baseline(df)
