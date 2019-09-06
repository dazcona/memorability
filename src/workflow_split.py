# import libraries
import os
import numpy as np
import sys
import config
from loader import data_load
from sklearn.model_selection import train_test_split
from tfidf import fit_and_transform_text
from embeddings import train_embeddings_network
from features import get_features_from_last_layer_pretrained_nn
from modelling import train_predict
from evaluate import evaluate_spearman
from plotting import plot_true_vs_predicted
from timeit import default_timer


## START

start = default_timer()

## SUM OF FEATURES WEIGHTS MUST BE EQUAL TO 1
print(config.FEATURES_WEIGHTS.values())
print(sum(config.FEATURES_WEIGHTS.values()))
print(sum(config.FEATURES_WEIGHTS.values()) == 1)

assert round(sum(config.FEATURES_WEIGHTS.values())) == 1, 'ERROR: Sum of feature weights must be equal to 1. Fix your config file'

## LOGDIR

print('[INFO] Creating log {} and checkpoints dir {}...'.format(config.RUN_LOG_DIR, config.RUN_CHECKPOINT_DIR))
os.mkdir(config.RUN_LOG_DIR)
os.mkdir(config.RUN_CHECKPOINT_DIR)
MAIN_LOG = os.path.join(config.RUN_LOG_DIR, 'mainlog.txt')

## SEED

np.random.seed(42)

## VALUES

with open(MAIN_LOG, 'w') as f:
    print("""MEMORABILITY 2019
SPLIT APPROACH
TARGET: {}
FEATURES_WEIGHTS: {}
FEATURES_ALGORITHM: {}
NUM_EPOCHS: {}""".format(config.TARGET,
    ', '.join([ '{} ({:.0%})'.format(feature, weight) for feature, weight in config.FEATURES_WEIGHTS.items()
                                                        if weight > 0 ]),
    ', '.join([ '{}: {}'.format(feature, config.FEATURES_ALGORITHM[feature])
                                        for feature, weight in config.FEATURES_WEIGHTS.items()
                                            if weight > 0 ]),
    config.NUM_EPOCHS), file=f)

## LOAD DATA

print('[INFO] Loading data...')
dataframe = data_load(config.FEATURES_WEIGHTS)

X = dataframe.drop(columns=config.TARGET_COLS)
Y = dataframe[config.TARGET]

## SPLIT DATA

test_size = 0.125 # 1,000 out of 8,000
print('[INFO] Splitting data between train ({:.2%}) and validation ({:.2%}) sets...'.format(1 - test_size, test_size))
X_train, X_val, y_train, y_val = train_test_split(
    X,
    Y,
    test_size=test_size,
    random_state=42)
print('[INFO] Number of training samples: {}'.format(len(X_train)))
print('[INFO] Number of validation samples: {}'.format(len(X_val)))

predictions = []

## FEATURES

for feature_name in config.FEATURES_WEIGHTS:

    if config.FEATURES_WEIGHTS[feature_name] > 0:

        ## DEEP LEARNING

        if feature_name == 'CAPTIONS' and config.ENCODING_ALGORITHM['CAPTIONS'] == 'EMBEDDINGS':

            ## ENCODING & MODELLING

            print('[INFO] Processing Embeddings...')
            fold = 0
            os.mkdir(config.RUN_LOG_FOLD_DIR.format(fold))
            predictions_features = train_embeddings_network(X_train['caption'], y_train,
                                                            X_val['caption'], y_val,
                                                            fold)
            predictions.append(predictions_features * config.FEATURES_WEIGHTS[feature_name])

        elif feature_name == 'PRE-TRAINED NN':

            pretrained_nn = config.PRE_TRAINED_NN

            X_train_features = get_features_from_last_layer_pretrained_nn(X_train['video'], config.DEV_FRAMES,
                                pretrained_nn, 'train')
            X_val_features = get_features_from_last_layer_pretrained_nn(X_val['video'], config.DEV_FRAMES,
                                pretrained_nn, 'val')

            ## MODELLING

            print('[INFO] Number of features: {:,}'.format(X_train_features.shape[1]))
            predictions_features = train_predict(X_train_features, y_train, X_val_features, pretrained_nn,
                                                 config.FEATURES_ALGORITHM[feature_name],
                                                 grid_search=config.GRID_SEARCH)
            predictions.append(predictions_features * config.FEATURES_WEIGHTS[feature_name])

        ## TRADITIONAL MACHINE LEARNING

        else:

            if feature_name == 'CAPTIONS': # TF-IDF

                # TRANSFORM CAPTIONS

                print('[INFO] Processing the captions and transforming them into numbers...')
                X_train_features, X_val_features = fit_and_transform_text(X_train['caption'], X_val['caption'])
                X_train_features = X_train_features.toarray()
                X_val_features = X_val_features.toarray()

            else:

                print('[INFO] Processing {} features...'.format(feature_name))
                X_train_features = X_train.filter(regex=('{}_*'.format(feature_name)))
                X_val_features = X_val.filter(regex=('{}_*'.format(feature_name)))

            ## MODELLING

            print('[INFO] Number of features: {:,}'.format(X_train_features.shape[1]))
            predictions_features = train_predict(X_train_features, y_train, X_val_features, feature_name,
                                                 config.FEATURES_ALGORITHM[feature_name],
                                                 grid_search=config.GRID_SEARCH)
            predictions.append(predictions_features * config.FEATURES_WEIGHTS[feature_name])

## EVALUATE

predictions = np.sum(predictions, axis=0)

print('[INFO] Evaluating the performance of the predictions...')
corr_coefficient, p_value = evaluate_spearman(y_val, predictions)
print('[INFO] Spearman Correlation Coefficient: {:.5f} (p-value {:.5f})'.format(corr_coefficient, p_value))

with open(MAIN_LOG, 'a') as f:
    print("""RESULTS: Spearman Correlation Coefficient: {:.5f} (p-value {:.5f})""".format(corr_coefficient, p_value), file=f)

## PLOT EVALUATION

plot_true_vs_predicted(y_val, predictions)

## END

end = default_timer() - start
minutes, seconds = divmod(end, 60)
print('[INFO] Execution duration: {:.2f} minutes {:.2f} seconds'.format(minutes, seconds))

with open(MAIN_LOG, 'a') as f:
    print('[INFO] Execution duration: {:.2f} minutes {:.2f} seconds'.format(minutes, seconds), file=f)
