# imports
import os
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNetCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import config


ALGORITHMS = {
    'Bayesian Ridge': {
        'model': BayesianRidge(),
        'params': {
            "alpha_1": [0.000001, 0.0001, 0.1],
            "alpha_2": [0.000001, 0.0001, 0.1],
            "lambda_1": [0.000001, 0.0001, 0.1],
            "lambda_2": [0.000001, 0.0001, 0.1],
            "normalize": [True, False],
        }
    },
    'SVM': {
        'model': SVR(gamma='scale'),
        'params': {
            "kernel": ['linear', 'rbf' ], # 'poly', 'sigmoid'
            "C": [0.1, 1.0, 100.0, 10000.0],
            "epsilon": [0.001, 1.0],
        }
    }
}
BEST_MODEL = {
    'short-term_memorability': {
        'C3D': BayesianRidge(lambda_2=1e-06, alpha_1=1e-06, alpha_2=0.1, normalize=False, lambda_1=0.1),
        'AESTHETICS': BayesianRidge(lambda_2=0.1, alpha_1=0.1, alpha_2=1e-06, normalize=True, lambda_1=1e-06),
        'HMP': SVR(kernel='linear', C=100.0, epsilon=0.001, gamma='scale'),
        'ColorHistogram': SVR(kernel='rbf', gamma='scale'),
        'LBP': BayesianRidge(lambda_2=0.1, alpha_1=0.1, alpha_2=1e-06, normalize=False, lambda_1=1e-06),
        'InceptionV3': BayesianRidge(),
        'CAPTIONS': SVR(gamma='scale'),
        'ResNet50': SVR(gamma='scale', kernel='rbf', C=0.1, epsilon=0.001),
        'ResNet152': SVR(kernel='rbf', C=0.1, epsilon=0.001, gamma='scale'),
        'ResNet152V2': SVR(gamma='scale'),
        'Emotions': BayesianRidge(),
        'Our_Aesthetics': BayesianRidge(),
    },
    'long-term_memorability' : {
        'C3D': BayesianRidge(lambda_2=0.0001, alpha_1=1e-06, alpha_2=0.1, normalize=False, lambda_1=0.1),
        'AESTHETICS': BayesianRidge(lambda_2=1e-06, alpha_1=1e-06, alpha_2=0.1, normalize=True, lambda_1=0.1),
        'HMP': SVR(kernel='linear', epsilon=0.001, C=100.0, gamma='scale'),
        'ColorHistogram': BayesianRidge(lambda_2=0.0001, alpha_1= 1e-06, alpha_2=0.1, normalize=True, lambda_1=0.1),
        'LBP': BayesianRidge(lambda_2=0.0001, alpha_1= 0.1, alpha_2=1e-06, normalize=True, lambda_1=1e-06),
        'InceptionV3': BayesianRidge(),
        'CAPTIONS': SVR(gamma='scale'),
        'ResNet50': SVR(kernel='rbf', C=0.1, epsilon=0.001, gamma='scale'),
        'ResNet152': SVR(kernel='rbf', C=0.1, epsilon=0.001, gamma='scale'),
        'ResNet152V2': SVR(gamma='scale'),
        'Emotions': BayesianRidge(lambda_2=0.0001, alpha_1=1e-06, alpha_2=0.1, normalize=True, lambda_1=0.1),
        'Our_Aesthetics': BayesianRidge(),
    },
}
CV = 3


def train_predict(X_train, y_train, X_val, feature_name, method, grid_search=False):

    # Model
    if grid_search:
        print('[INFO] Performing Grid Search for {} on {}...'.format(method, feature_name))
        with open(os.path.join(config.RUN_LOG_DIR, 'mainlog.txt'), 'a') as f:
            print('[INFO] Performing Grid Search for {} on {}...'.format(method, feature_name), file=f)
        # Perform grid search
        model = GridSearchCV(
            ALGORITHMS[method]['model'],
            ALGORITHMS[method]['params'],
            cv=CV,
            n_jobs=-1,
        )
    else:
        print('[INFO] Initializing model for {}...'.format(feature_name))
        with open(os.path.join(config.RUN_LOG_DIR, 'mainlog.txt'), 'a') as f:
            print('[INFO] Initializing model for {}...'.format(feature_name), file=f)
        model = BEST_MODEL[config.TARGET][feature_name]

    # Fit model
    print('[INFO] Fitting model...')
    model.fit(X_train, y_train)

    if grid_search:
        # Best hyperparameters
        print("[INFO] best hyperparameters: {}".format(model.best_params_))
        with open(os.path.join(config.RUN_LOG_DIR, 'mainlog.txt'), 'a') as f:
            print("""GRID SEARCH: Best hyperparameters for {}: {})""".format(method, model.best_params_), file=f)
    else:
        print("[INFO] Parameters: {}".format(model.get_params()))
        with open(os.path.join(config.RUN_LOG_DIR, 'mainlog.txt'), 'a') as f:
            print("""MODEL: Parameters for {}: {})""".format(method, model.get_params()), file=f)

    # Predict
    print('[INFO] Predict new values...')
    predicted = model.predict(X_val)

    return predicted


def fit_predict(X_train_val, Y_train_val, X_test, feature_name):

    print('[INFO] Loading model...')
    model = BEST_MODEL[config.TARGET][feature_name]

    # Fit model
    print('[INFO] Fitting model...')
    model.fit(X_train_val, Y_train_val)

    print('[INFO] Predicting values...')
    predicted = model.predict(X_test)

    return predicted


if __name__ == "__main__":
    n_samples, n_features = 10, 5
    rng = np.random.RandomState(0)
    y_train = rng.randn(n_samples)
    X_train = rng.randn(n_samples, n_features)
    X_test = rng.randn(n_samples, n_features)
    predictions = train_predict(X_train, y_train, X_test)
    print(predictions)
