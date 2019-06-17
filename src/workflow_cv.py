import numpy as np

## VALUES

# Column to predict
TARGET = 'short-term_memorability'
# Target columns
TARGET_COLS = [ 'short-term_memorability', 'nb_short-term_annotations', 'long-term_memorability', 'nb_long-term_annotations' ]
# Dictionary that indicates which data sources to use in the model
FEATURES_DICT = {
    'CAPTIONS': True,
    'C3D': True,
    'AESTHETICS': True,
    'HMP': False,
}
FEATURES_WEIGHTS = {
    'CAPTIONS': 0.6,
    'C3D': 0.3,
    'AESTHETICS': 0.1,
    'HMP': 0,
}
# Algorithm
ALGORITHM = 'Bayesian Ridge'
# Number of folds for Cross-Validation
NFOLDS = 10

## LOAD DATA

from loader import data_load

print('[INFO] Loading data with captions...')
dataframe = data_load(FEATURES_DICT)

X = dataframe.drop(columns=TARGET_COLS)
Y = dataframe[TARGET]

import sys

if len(X.columns) <= 1:
    print('[FAIL] Number of features fewer than one, please load features to the dataset')
    sys.exit(1)

## CROSS VALIDATION

from sklearn.model_selection import KFold

kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

k = 0

scores = []

for train_index, val_index in kf.split(X):

    ## SPLIT

    print('[INFO] {}. Cross-Validation fold: {}'.format(k + 1, k + 1))
    k += 1

    X_train, X_val = X.iloc[train_index, :], X.iloc[val_index, :]
    y_train, y_val = Y.iloc[train_index], Y.iloc[val_index]

    from processing import fit_and_transform_text
    from modelling import train_predict

    predictions = []

    ## PRE-COMPUTED FEATURES

    for feature_name in [ 
        'CAPTIONS',
        'C3D', 
        'AESTHETICS', 
        'HMP',
        ]:

        if FEATURES_DICT[feature_name]:

            ## GET AND PROCESS DATA

            if feature_name == 'CAPTIONS':

                print('[INFO] Processing the captions and transforming them into numbers...')
                X_train_features, X_val_features = fit_and_transform_text(X_train['caption'], X_val['caption'])
                X_train_features = X_train_features.toarray()
                X_val_features = X_val_features.toarray()

            else:

                print('[INFO] Processing {} features...'.format(feature_name))
                X_train_features = X_train.filter(regex=('{}_*'.format(feature_name)))
                X_val_features = X_val.filter(regex=('{}_*'.format(feature_name)))

            ## MODELLING

            predictions_features = train_predict(X_train_features, y_train, X_val_features, method=ALGORITHM)
            predictions.append(predictions_features * FEATURES_WEIGHTS[feature_name])
    
    ## EVALUATE

    predictions = np.sum(predictions, axis=0)

    from evaluate import evaluate_spearman

    print('[INFO] Evaluating the performance of the predictions...')
    corr_coefficient, p_value = evaluate_spearman(y_val, predictions)
    print('[INFO] Spearman Correlation Coefficient: {:.4f} (p-value {:.4f})'.format(corr_coefficient, p_value))
    scores.append(corr_coefficient)

print('-' * 80)
print('[INFO] MEAN of Spearman Correlation Coefficients thoughout the {} folds: {:.5f}'.format(NFOLDS, np.mean(scores)))
print('-' * 80)

# SVM
# [INFO] MEAN of Spearman Correlation Coefficients thoughout the 10 folds: 0.41422
# Bayesian Ridge
# [INFO] MEAN of Spearman Correlation Coefficients thoughout the 10 folds: 0.45983