import numpy as np
import scipy as scp
import pandas as pd
import seaborn as sn
from matplotlib import pyplot
from hyperopt import fmin, space_eval, hp, tpe, STATUS_OK

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

models = {
    'KNN'                : KNeighborsClassifier(),
    'SVM'         : SVC(kernel="linear", C=0.025),
    'Gaussian Process'   : GaussianProcessClassifier(1.0 * RBF(1.0)),
    'Decision Tree'      : DecisionTreeClassifier(max_depth=5),
    'Random Forest'      : RandomForestClassifier(max_depth=5),
    'Neural Network'     : MLPClassifier(alpha=1, max_iter=1000),
    'AdaBoost'           : AdaBoostClassifier(),
    'Naive Bayes'        : GaussianNB(),
    'QDA'                : QuadraticDiscriminantAnalysis(),
    'Logistic Regression': LogisticRegression(max_iter=500),
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
}

category_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

def preprocess(df: pd.DataFrame):

    le = LabelEncoder()
    for column in category_columns:
        df[column] = le.fit_transform(df[column])

    return df

def find_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, float]:

    X_train = train_df.drop('HeartDisease', axis=1).values
    X_test = test_df.drop('HeartDisease', axis=1).values

    y_train = train_df['HeartDisease'].values
    y_test = test_df['HeartDisease'].values

    scaler = StandardScaler()
    results = {}

    for name,  model in models.items():
        
        scaler.fit_transform(X_train)
        scaler.transform(X_test)

        model.fit(X=X_train, y=y_train)
        y_pred = model.predict(X_test)

        score = accuracy_score(y_test, y_pred)
        results[name] = score

    return results

def create_objective(classifier_name: str, df: pd.DataFrame, scale_values: bool = False):

    X = df.drop('HeartDisease', axis=1).values
    y = df['HeartDisease'].values

    def objective(search_space):

        classifier = classifiers[classifier_name]

        try:
            model = classifier(**search_space, random_state=42)
        except:
            model = classifier(**search_space)

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

def optimize_hyperparams(classifier_name: str, train_df: pd.DataFrame, max_evals: int):
    optimized_params = fmin(
        fn=create_objective(classifier_name, train_df),
        space=search_spaces[classifier_name],
        algo=tpe.suggest,
        max_evals=max_evals
    )

    return space_eval(search_spaces[classifier_name], optimized_params)

if __name__ == '__main__':
    df = pd.read_csv("./heart_failure.csv")
    df = preprocess(df)
    find_baseline(df)
