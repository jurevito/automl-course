import numpy as np
import scipy as scp
import pandas as pd
import seaborn as sn
from matplotlib import pyplot
from hyperopt import fmin, space_eval, hp, tpe, STATUS_OK, STATUS_FAIL, Trials
from typing import Protocol, Dict, List
from scipy.stats import wilcoxon

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import KFold

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

class ScikitModel(Protocol):
    def fit(self, X, y, sample_weight=None): ...
    def predict(self, X) -> np.ndarray: ...
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
    'LDA'                : LinearDiscriminantAnalysis(),
    'Bagging Classifier' : BaggingClassifier(random_state=42),
    'Gradient Boosting'  : GradientBoostingClassifier(random_state=42),
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
    'LDA'                : LinearDiscriminantAnalysis,
    'Bagging Classifier' : BaggingClassifier,
    'Gradient Boosting'  : GradientBoostingClassifier,
}

search_spaces = {
    'Random Forest': {
        'n_estimators'            : hp.randint('n_estimators', 50, 150),
        'criterion'               : hp.choice('criterion', ['gini', 'entropy', 'log_loss']),
        'max_depth'               : hp.randint('max_depth', 10, 200),
        'min_samples_split'       : hp.randint('min_samples_split', 2, 3),
        'min_samples_leaf'        : hp.randint('min_samples_leaf', 1, 2),
        'min_weight_fraction_leaf': hp.uniform('min_weight_fraction_leaf', 0, 0.05),
        'max_features'            : hp.choice('max_features', ['sqrt', 'log2']),
    },

    'Logistic Regression': {
        'penalty' : hp.choice('penalty', ['l2']),
        'C'       : hp.uniform('C', 0.95, 1.0),
        'solver'  : hp.choice('solver', ['newton-cg', 'lbfgs', 'liblinear']),
        'max_iter': hp.choice('max_iter', [1000])
    },

    'AdaBoost': {
        'n_estimators' : hp.randint('n_estimators', 20, 500),
        'learning_rate': hp.uniform('learning_rate', 0.8, 2.0),
        'algorithm'    : hp.choice('algorithm', ['SAMME', 'SAMME.R'])
    },

    'Neural Network': {
        'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [(100,), (50,), (200,), (150,)]),
        'activation'        : hp.choice('activation', ['identity', 'logistic', 'tanh', 'relu']),
        'solver'            : hp.choice('solver', ['lbfgs', 'sgd', 'adam']),
        'alpha'             : hp.uniform('alpha', 0, 1e-3),
        'learning_rate'     : hp.choice('learning_rate', ['constant', 'invscaling', 'adaptive']),
        'max_iter'          : hp.choice('max_iter', [100])
    },

    'Decision Tree': {
        'criterion'         : hp.choice('criterion', ['gini', 'entropy', 'log_loss']),
        'max_depth'         : hp.randint('max_depth', 2, 20),
        'min_samples_split' : hp.randint('min_samples_split', 2, 3),
        'min_samples_leaf'  : hp.randint('min_samples_leaf', 1, 2),
        'max_features'      : hp.choice('max_features', ['auto', 'sqrt', 'log2', None])
    },

    'Gaussian Process': {
        'kernel'               : hp.choice('kernel', [1.0 * RBF(1.0)]),
        'n_restarts_optimizer' : hp.randint('n_restarts_optimizer', 0, 5),
        'warm_start'           : hp.choice('warm_start', [False, True]),
        'max_iter_predict'     : hp.choice('max_iter_predict', [250]),
        'n_jobs'               : hp.choice('n_jobs', [-1]),
    },

    'SVM': {
        'C'                     : hp.uniform('C', 0.95, 1.0),
        'kernel'                : hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
        'gamma'                 : hp.choice('gamma', ['scale', 'auto']),
        'break_ties'            : hp.choice('break_ties', [False, True]),
        'max_iter'              : hp.choice('max_iter', [1000]),
    },

    'KNN': {
        'n_neighbors'       : hp.randint('n_neighbors', 2, 8),
        'weights'           : hp.choice('weights', ['uniform', 'distance']),
        'algorithm'         : hp.choice('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
        'leaf_size'         : hp.randint('leaf_size', 10, 40),
        'p'                 : hp.choice('p', [1, 2]),
        'n_jobs'            : hp.choice('n_jobs', [-1]),
    },

    'Naive Bayes': {
        'var_smoothing'     : hp.uniform('var_smoothing', 1e-11, 1e-8)
    },

    'QDA': {
        'reg_param'         : hp.uniform('reg_param', 0.0, 0.4),
    },

    'LDA': {
        'solver'   : hp.choice('solver', ['svd', 'lsqr', 'eigen']),
        'shrinkage': hp.choice('shrinkage', [None, 'auto']),
        'tol'      : hp.uniform('tol', 1.0e-5, 1.0e-1)
    },

    'Bagging Classifier': {
        'n_estimators'      : hp.randint('n_estimators', 5, 50),
        'max_samples'       : hp.uniform('max_samples', 0.1, 1.0),
        'max_features'      : hp.uniform('max_features', 0.5, 1.0),
        'bootstrap'         : hp.choice('bootstrap', [True, False]),
        'bootstrap_features': hp.choice('bootstrap_features', [True, False]),
        'warm_start'        : hp.choice('warm_start', [False, True]),
        'n_jobs'            : hp.choice('n_jobs', [-1]),
    },

    'Gradient Boosting': {
        'loss'                     : hp.choice('loss', ['log_loss', 'exponential']),
        'learning_rate'            : hp.uniform('learning_rate', 0.01, 0.2),
        'n_estimators'             : hp.randint('n_estimators', 10, 200),
        'subsample'                : hp.uniform('subsample', 0.5, 1.0),
        'criterion'                : hp.choice('criterion', ['friedman_mse', 'squared_error']),
        'min_samples_split'        : hp.uniform('min_samples_split', 0.01, 0.25),
        'min_samples_leaf'        : hp.randint('min_samples_leaf', 1, 2),
        'max_depth'                : hp.randint('max_depth', 2, 15),
        'max_features'             : hp.choice('max_features', [None, 'auto','sqrt', 'log2']),
    },
}

category_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

def one_hot_encode(X_train, X_test):

    transformer = make_column_transformer(
        (OneHotEncoder(), [2, 6, 10]),
        remainder='passthrough'
    )

    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)

    return X_train, X_test

def find_baseline(df: pd.DataFrame) -> Dict[str, float]:
    X = df.drop('HeartDisease', axis=1).values
    y = df['HeartDisease'].values

    scaler = StandardScaler()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, model in models.items():
        scores = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            X_train, X_test = one_hot_encode(X_train, X_test)
            scaler.fit_transform(X_train)
            scaler.transform(X_test)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            score = accuracy_score(y_test, y_pred)
            scores.append(score)

        results[name] = sum(scores) / len(scores)

    return results

def create_objective(name: str, df: pd.DataFrame):
    X = df.drop('HeartDisease', axis=1).values
    y = df['HeartDisease'].values

    def objective(search_space):

        classifier = classifiers[name]

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

            X_train, X_test = one_hot_encode(X_train, X_test)
            scaler.fit_transform(X_train)
            scaler.transform(X_test)

            try:
                model.fit(X_train, y_train)
            except:
                print('Failed a trial.')
                print(search_space)
                return {'loss': 1.0, 'status': STATUS_FAIL}

            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            scores.append(score)

        avg_score = sum(scores) / len(scores)

        return {'loss': 1 - avg_score, 'status': STATUS_OK}

    return objective


def unpack_vals(values: dict):
    vals = {}
    for key, value in values['misc']['vals'].items():
        if value:
            vals[key] = value[0]

    return vals

def optimize_hyperparams(name: str, df: pd.DataFrame, max_evals: int, scale_values: bool = True):

    trials = Trials()

    optimized_params = fmin(
        fn=create_objective(name, df),
        space=search_spaces[name],
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(42),
    )

    best_params = space_eval(search_spaces[name], optimized_params)
    losses = []
    params = []
    statuses = []

    # Save losses and corresponding hyperparameter configurations.
    for trial in trials:
        losses.append(trial['result']['loss'])
        params.append(space_eval(search_spaces[name], unpack_vals(trial)))
        statuses.append(trial['result']['status'])

    return best_params, losses, params, statuses

def compare_with_baseline(baseline: str, optimization_results: dict, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:

    X_train = train_df.drop('HeartDisease', axis=1).values
    X_test = test_df.drop('HeartDisease', axis=1).values

    y_train = train_df['HeartDisease'].values
    y_test = test_df['HeartDisease'].values

    X_train, X_test = one_hot_encode(X_train, X_test)
    scaler = StandardScaler()
    scaler.fit_transform(X_train)
    scaler.transform(X_test)

    # Train a baseline model.
    model: ScikitModel = models[baseline]
    model.fit(X_train, y_train)
    y_pred_baseline = model.predict(X_test)
    score_baseline = accuracy_score(y_test, y_pred_baseline)

    results = {}

    for name in classifiers:
        classifier = classifiers[name]
        params = optimization_results[name]['best']

        try:
            model: ScikitModel = classifier(**params, random_state=42)
        except:
            model: ScikitModel = classifier(**params)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)

        stat, p_value = 1.0, 1.0
        if not (y_pred_baseline==y_pred).all():
            stat, p_value = wilcoxon(y_pred_baseline, y_pred)

        results[name] = {
            'stat': stat,
            'p_value': p_value,
            'score': score,
        }

    return score_baseline, results

def find_min_budget(baseline: str, train_df: pd.DataFrame, test_df: pd.DataFrame, optimization_results: dict):

    X_train = train_df.drop('HeartDisease', axis=1).values
    X_test = test_df.drop('HeartDisease', axis=1).values

    y_train = train_df['HeartDisease'].values
    y_test = test_df['HeartDisease'].values

    X_train, X_test = one_hot_encode(X_train, X_test)
    scaler = StandardScaler()
    scaler.fit_transform(X_train)
    scaler.transform(X_test)

    # Train a baseline model.
    model: ScikitModel = models[baseline]
    model.fit(X_train, y_train)
    y_pred_baseline = model.predict(X_test)
    score_baseline = accuracy_score(y_test, y_pred_baseline)

    for name in classifiers:
        optimization_results[name]['min_budget'] = len(optimization_results[name]['losses'])

        for step, (params, status) in enumerate(zip(optimization_results[name]['params'], optimization_results[name]['statuses'])):
            if status != 'ok':
                continue

            classifier = classifiers[name]

            try:
                model: ScikitModel = classifier(**params, random_state=42)
            except:
                model: ScikitModel = classifier(**params)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)

            if score > score_baseline:
                optimization_results[name]['min_budget'] = step+1
                break



if __name__ == '__main__':
    df = pd.read_csv("./heart_failure.csv")
