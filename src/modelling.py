# imports
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNetCV, LassoCV
from sklearn.ensemble import RandomForestRegressor

def train_predict(X_train, y_train, X_val, method='SVM'):
    
    # Create model
    print('[INFO] Initializing model...')
    if method == 'SVM':
        model = SVR(gamma='scale')
    elif method == 'SVM Gaussian':
        model = SVR(kernel='rbf', gamma='scale')
    elif method == 'Linear Regression':
        model = LinearRegression()
    elif method == 'Bayesian Ridge':
        model = BayesianRidge()
    elif method == 'Elastic Net':
        model = ElasticNetCV(cv=5)
    elif method == 'Lasso':
        model = LassoCV(cv=5)
    elif method == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100)
    else:
        raise Exception('Learning method {} not found!'.format(method))
    
    # Fit model
    print('[INFO] Fitting model...')
    model.fit(X_train, y_train)
    
    # Predict
    print('[INFO] Predict new values...')
    predicted = model.predict(X_val)

    return predicted


if __name__ == "__main__":
    n_samples, n_features = 10, 5
    rng = np.random.RandomState(0)
    y_train = rng.randn(n_samples)
    X_train = rng.randn(n_samples, n_features)
    X_test = rng.randn(n_samples, n_features)
    predictions = train_predict(X_train, y_train, X_test)
    print(predictions)
