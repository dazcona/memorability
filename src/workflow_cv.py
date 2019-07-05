# import libraries
import os
import numpy as np
import sys
import config
from loader import data_load, load_pretrained_word_vectors
from sklearn.model_selection import KFold
from tfidf import fit_and_transform_text
from embeddings import train_embeddings_network
from modelling import train_predict
from evaluate import evaluate_spearman


## LOGDIR

print('[INFO] Creating log {} and checkpoints dir {}...'.format(config.RUN_LOG_DIR, config.RUN_CHECKPOINT_DIR))
os.mkdir(config.RUN_LOG_DIR)
os.mkdir(config.RUN_CHECKPOINT_DIR)
MAIN_LOG = os.path.join(config.RUN_LOG_DIR, 'mainlog.txt')

## SEED

np.random.seed(42)

## VALUES

with open(MAIN_LOG, 'w') as f:
    print("""TARGET: {}
FEATURES: {}
FEATURES_WEIGHTS: {}
CAPTIONS_ALGORITHM: {} (if captions are used)
NUM_EPOCHS: {}
ALGORITHM: {} (if pre-computed features are used)
NFOLDS: {}""".format(config.TARGET, 
    ','.join([ feature for feature, added in config.FEATURES_DICT.items() if added ]), 
    ','.join([ '{} ({:.0%})'.format(feature, weight) for feature, weight in config.FEATURES_WEIGHTS.items() if weight > 0 ]), 
    config.CAPTIONS_ALGORITHM,
    config.NUM_EPOCHS,
    config.ALGORITHM,
    config.NFOLDS), file=f)

## LOAD DATA

print('[INFO] Loading data with captions...')
dataframe = data_load(config.FEATURES_DICT)

X = dataframe.drop(columns=config.TARGET_COLS)
Y = dataframe[config.TARGET]

word_vectors = load_pretrained_word_vectors()

if len(X.columns) <= 1:
    print('[FAIL] Number of features fewer than one, please load features to the dataset')
    sys.exit(1)

## CROSS VALIDATION

kf = KFold(n_splits=config.NFOLDS, shuffle=True, random_state=42)

k = 0

scores = []

for train_index, val_index in kf.split(X):

    fold = fold

    ## SPLIT

    print('[INFO] {}. Cross-Validation fold: {}'.format(fold, fold))

    X_train, X_val = X.iloc[train_index, :], X.iloc[val_index, :]
    y_train, y_val = Y.iloc[train_index], Y.iloc[val_index]

    os.mkdir(config.RUN_LOG_FOLD_DIR.format(fold))

    predictions = []

    ## PRE-COMPUTED FEATURES

    for feature_name in config.FEATURES_DICT:

        if config.FEATURES_DICT[feature_name]:

            ## DEEP LEARNING

            if feature_name == 'CAPTIONS' and config.CAPTIONS_ALGORITHM == 'EMBEDDINGS':

                print('[INFO] Processing Embeddings...')
                predictions_features = train_embeddings_network(X_train['caption'], y_train, X_val['caption'], y_val, word_vectors, fold)
                predictions.append(predictions_features)

            ## TRADITIONAL MACHINE LEARNING

            else:

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

                print('[INFO] Number of features: {}'.format(X_train_features.shape[1]))
                predictions_features = train_predict(X_train_features, y_train, X_val_features, method=config.ALGORITHM)
                predictions.append(predictions_features * FEATURES_WEIGHTS[feature_name])
    
    ## EVALUATE

    predictions = np.sum(predictions, axis=0)

    print('[INFO] Evaluating the performance of the predictions...')
    corr_coefficient, p_value = evaluate_spearman(y_val, predictions)
    print('[INFO] Spearman Correlation Coefficient: {:.4f} (p-value {:.4f})'.format(corr_coefficient, p_value))
    scores.append(corr_coefficient)

    with open(MAIN_LOG, 'a') as f:
        print("""FOLD {} Cross-Validation. Spearman Correlation Coefficient: {:.4f} (p-value {:.4f})""".format(fold, corr_coefficient, p_value), file=f)

    ## NEXT

    k += 1

print('-' * 80)
print('[INFO] MEAN of Spearman Correlation Coefficients throughout the {} folds: {:.5f}'.format(config.NFOLDS, np.mean(scores)))
print('-' * 80)

with open(MAIN_LOG, 'a') as f:
    print("""MEAN of Spearman Correlation Coefficients throughout the {} folds: {:.5f}""".format(config.NFOLDS, np.mean(scores)), file=f)
