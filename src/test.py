# import libraries
import os
import numpy as np
import sys
import config
from loader import data_load

## SEED

np.random.seed(42)

## LOAD DATA

print('[INFO] Loading all features...')
FEATURES_WEIGHTS = {
    'C3D': 1,
    'AESTHETICS': 1,
    'HMP': 1,
    'ColorHistogram': 1,
    'LBP': 1,
    'InceptionV3': 1,
    'CAPTIONS': 1,
    'PRE-TRAINED NN': 1,
    'FINE-TUNED NN': 1,
    'Emotions': 1,
    'Our_Aesthetics': 1,
}
dataframe = data_load(FEATURES_WEIGHTS, config.TEST_GROUNDTRUTH, config.TEST_CAPTIONS, config.TEST_FEATURES_PATH)

X_test = dataframe.drop(columns=config.TARGET_COLS)

print('[INFO] Number of test samples: {}'.format(len(X_test)))

## FEATURES

## CAPTIONS, EMBEDDINGS, DEEP LEARNING

print('[INFO] Processing captions with embeddings...')
predictions_features = predict_with_embeddings(X_val['caption'])

# preds_filename = 'final_predictions/{}_captions_embeddings'.format(config.TARGET)
#np.save(preds_filename, predictions_features)

# PRE-TRAINED CNN AS FEATURE EXTRACTOR

pretrained_nn = config.PRE_TRAINED_NN
print('[INFO] {} as feature extractor...'.format(pretrained_nn))
X_test_features = get_features_from_last_layer_pretrained_nn(X_test['video'], config.TEST_FRAMES,
                    pretrained_nn, 'test')
# predict...

#preds_filename = 'final_predictions/{}_pretrained_{}'.format(config.TARGET, pretrained_nn)

