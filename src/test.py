# import libraries
import os
import numpy as np
import sys
import config
from loader import data_load
from embeddings import fit_predict_with_embeddings
from pretrained_cnn import get_features_from_last_layer_pretrained_nn
from modelling import fit_predict
from aesthetics import videos_to_our_aesthetics
from emotions import get_emotions_test_data
from timeit import default_timer

## START

start = default_timer()

## SEED

np.random.seed(42)

## LOAD DATA

FEATURES_WEIGHTS = { 
    'C3D': 1,
    'AESTHETICS': 0,
    'HMP': 0,
    'ColorHistogram': 0,
    'LBP': 1,
    'InceptionV3': 0,
    'CAPTIONS': 1,
    'PRE-TRAINED NN': 1,
    'FINE-TUNED NN': 1,
    'Emotions': 1,
    'Our_Aesthetics': 1,
}
print('[INFO] Predicting {}-term memorability'.format(config.TARGET_SHORT_NAME))
print('[INFO] Loading training and validation data...')
dataframe = data_load(
    FEATURES_WEIGHTS, 
    config.DEV_GROUNDTRUTH, 
    config.DEV_CAPTIONS, 
    config.DEV_FEATURES_PATH,
    'train_val',
)
X = dataframe.drop(columns=config.TARGET_COLS)
Y = dataframe[config.TARGET]

print('[INFO] Loading test data...')
X_test = data_load(
    FEATURES_WEIGHTS, 
    config.TEST_GROUNDTRUTH, 
    config.TEST_CAPTIONS, 
    config.TEST_FEATURES_PATH,
    'test',
)

print('[INFO] Number of training and validation samples: {:,}'.format(len(X)))
print('[INFO] Number of test samples: {:,}'.format(len(X_test)))

## FEATURES

## CAPTIONS, EMBEDDINGS, DEEP LEARNING
# Fit and predict
print('[INFO] Processing CAPTIONS with EMBEDDINGS...')
predictions_features = fit_predict_with_embeddings(X_test['caption'])
# Save predictions
preds_filename = '{}/{}_captions_embeddings'.format(config.PREDICTIONS_TEST, config.TARGET)
np.save(preds_filename, predictions_features)

# PRE-TRAINED CNN AS FEATURE EXTRACTOR

pretrained_nn = config.PRE_TRAINED_NN
print('[INFO] {} as feature extractor...'.format(pretrained_nn))
# Get features
X_features = get_features_from_last_layer_pretrained_nn(X['video'], config.DEV_FRAMES,
                pretrained_nn, 'train_val')
X_test_features = get_features_from_last_layer_pretrained_nn(X_test['video'], config.TEST_FRAMES, 
                    pretrained_nn, 'test')
# Fit and predict
predictions_features = fit_predict(X_features, Y, X_test_features, pretrained_nn)
# Save predictions
preds_filename = '{}/{}_pretrained_{}'.format(config.PREDICTIONS_TEST, config.TARGET, pretrained_nn)
np.save(preds_filename, predictions_features)

# OUR PRE-COMPUTED AESTHETICS BY INSIGHT

print('[INFO] Processing OUR AESTHETICS test features and applying a traditional ML approach...')
# Get features
X_features = videos_to_our_aesthetics(X['video'], config.OUR_AESTHETICS_DEV)
X_test_features = videos_to_our_aesthetics(X_test['video'], config.OUR_AESTHETICS_TEST)
# Fit and predict
predictions_features = fit_predict(X_features, Y, X_test_features, 'Our_Aesthetics')
# Save predictions
preds_filename = '{}/{}_our_aesthetics'.format(config.PREDICTIONS_TEST, config.TARGET)
np.save(preds_filename, predictions_features)

## OUR PRE-COMPUTED EMOTION FEATURES

print('[INFO] Processing EMOTION test features and applying a traditional ML approach...')
# Get features
X_features = get_emotions_test_data(X['video'], config.EMOTIONS_DEV, 'train_val')
X_test_features = get_emotions_test_data(X_test['video'], config.EMOTIONS_TEST, 'test')
# Fit and predict
predictions_features = fit_predict(X_features, Y, X_test_features, 'Emotions')
# Save predictions
preds_filename = '{}/{}_emotions'.format(config.PREDICTIONS_TEST, config.TARGET)
np.save(preds_filename, predictions_features)

## PRE-COMPUTED FEATURES: 'C3D', 'LBP'

for FEATURE in [ 'C3D', 'LBP' ]: 

    print('[INFO] Processing {} test features and applying a traditional ML approach...'.format(FEATURE))
    # Get features
    X_features = X.filter(regex=('{}_*'.format(FEATURE)))
    X_test_features = X_test.filter(regex=('{}_*'.format(FEATURE)))
    # Fit and predict
    predictions_features = fit_predict(X_features, Y, X_test_features, FEATURE)
    # Save predictions
    preds_filename = '{}/{}_{}'.format(config.PREDICTIONS_TEST, config.TARGET, FEATURE)
    np.save(preds_filename, predictions_features)

## END

end = default_timer() - start
minutes, seconds = divmod(end, 60)
print('[INFO] Execution duration: {:.2f} minutes {:.2f} seconds'.format(minutes, seconds))