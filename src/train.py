# import libraries
import os
import numpy as np
import sys
import config
from loader import data_load
from sklearn.model_selection import train_test_split
from tfidf import fit_and_transform_text
from embeddings import train_embeddings_network
from pretrained_cnn import get_features_from_last_layer_pretrained_nn
from modelling import train_predict
from evaluate import evaluate_spearman
from plotting import plot_true_vs_predicted
from fine_tuning import train_fine_tuned_cnn
from emotions import train_emotions, get_emotions_data
from aesthetics import train_our_aesthetics
from timeit import default_timer
from sms import send


## START

start = default_timer()

## SUM OF FEATURES WEIGHTS MUST BE EQUAL TO 1
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
TARGET: {}
FEATURES_WEIGHTS: {}
FEATURES_ALGORITHM: {}""".format(config.TARGET,
    ', '.join([ '{} ({:.0%})'.format(feature, weight) for feature, weight in config.FEATURES_WEIGHTS.items()
                                                        if weight > 0 ]),
    ', '.join([ '{}: {}'.format(feature, config.FEATURES_ALGORITHM[feature])
                                        for feature, weight in config.FEATURES_WEIGHTS.items()
                                            if weight > 0 ]),
), file=f)

## LOAD DATA

print('[INFO] Loading data...')
dataframe = data_load(config.FEATURES_WEIGHTS, config.DEV_GROUNDTRUTH, config.DEV_CAPTIONS, config.DEV_FEATURES_PATH)

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

            print('[INFO] Processing captions with embeddings...')
            preds_filename = 'predictions/{}_captions_embeddings'.format(config.TARGET)
            
            fold = 0
            os.mkdir(config.RUN_LOG_FOLD_DIR.format(fold))
            predictions_features = train_embeddings_network(X_train['caption'], y_train,
                                                            X_val['caption'], y_val,
                                                            fold)
            
            print('[INFO] Saving captions with embeddings predictions...')
            np.save(preds_filename, predictions_features)
            predictions.append(predictions_features * config.FEATURES_WEIGHTS[feature_name])

        elif feature_name == 'PRE-TRAINED NN':

            # PRE-TRAINED CNN AS FEATURE EXTRACTOR

            pretrained_nn = config.PRE_TRAINED_NN
            print('[INFO] {} as feature extractor...'.format(pretrained_nn))
            preds_filename = 'predictions/{}_pretrained_{}'.format(config.TARGET, pretrained_nn)

            X_train_features = get_features_from_last_layer_pretrained_nn(X_train['video'], config.DEV_FRAMES,
                                pretrained_nn, 'train')
            X_val_features = get_features_from_last_layer_pretrained_nn(X_val['video'], config.DEV_FRAMES,
                                pretrained_nn, 'val')

            ## MODELLING

            print('[INFO] Number of features: {:,}'.format(X_train_features.shape[1]))
            predictions_features = train_predict(X_train_features, y_train, X_val_features, pretrained_nn,
                                                 config.FEATURES_ALGORITHM[feature_name],
                                                 grid_search=config.GRID_SEARCH)
            
            print('[INFO] Saving predictions from features extracted from pre-trained CNN...')
            np.save(preds_filename, predictions_features)
            predictions.append(predictions_features * config.FEATURES_WEIGHTS[feature_name])

        elif feature_name == 'FINE-TUNED NN':

            ## FINE-TUNED CNN

            print('[INFO] Fine tuning our own network...')
            preds_filename = 'predictions/{}_fine_tuned'.format(config.TARGET)

            predictions_features = train_fine_tuned_cnn(X_train['video'], X_val['video'], 
                                                    y_train, y_val, config.DEV_FRAMES)

            print('[INFO] Saving predictions from our own fine-tuned network...')
            np.save(preds_filename, predictions_features)
            predictions.append(predictions_features * config.FEATURES_WEIGHTS[feature_name])

        elif feature_name == 'Emotions' and config.EMOTIONS_NN:

            ## EMOTIONS FEATURES

            print('[INFO] Using our own emotion features and apply a DL approach...')
            preds_filename = 'predictions/{}_emotions_DL'.format(config.TARGET)

            predictions_features = train_emotions(X_train['video'], X_val['video'], 
                                                    y_train, y_val, config.EMOTIONS_DEV)

            print('[INFO] Saving predictions from our own emotion features...')
            np.save(preds_filename, predictions_features)
            predictions.append(predictions_features * config.FEATURES_WEIGHTS[feature_name])

        ## TRADITIONAL MACHINE LEARNING

        else:

            if feature_name == 'CAPTIONS': # TF-IDF

                # TRANSFORM CAPTIONS

                print('[INFO] Processing the captions and transforming them into numbers...')
                preds_filename = 'predictions/{}_captions_tfidf'.format(config.TARGET)

                X_train_features, X_val_features = fit_and_transform_text(X_train['caption'], X_val['caption'])
                X_train_features = X_train_features.toarray()
                X_val_features = X_val_features.toarray()

            elif feature_name == 'Our_Aesthetics': # OUR PRE-COMPUTED AESTHETICS BY INSIGHT

                print('[INFO] Extracting our own aesthetics features...')
                preds_filename = 'predictions/{}_our_aesthetics'.format(config.TARGET)

                X_train_features, X_val_features = train_our_aesthetics(X_train['video'], X_val['video'], 
                                                                        config.OUR_AESTHETICS_DEV)

            elif feature_name == 'Emotions' and not config.EMOTIONS_NN:

                print('[INFO] Extracting our own emotion features and apply a traditional ML approach...')
                preds_filename = 'predictions/{}_emotions_ML'.format(config.TARGET)

                X_train_features, X_val_features = get_emotions_data(X_train['video'], X_val['video'], 
                                                                        config.EMOTIONS_DEV)

            else:

                print('[INFO] Processing {} features...'.format(feature_name))
                preds_filename = 'predictions/{}_{}'.format(config.TARGET, feature_name)

                X_train_features = X_train.filter(regex=('{}_*'.format(feature_name)))
                X_val_features = X_val.filter(regex=('{}_*'.format(feature_name)))

            ## MODELLING

            print('[INFO] Number of features: {:,}'.format(X_train_features.shape[1]))
            predictions_features = train_predict(X_train_features, y_train, X_val_features, feature_name,
                                                 config.FEATURES_ALGORITHM[feature_name],
                                                 grid_search=config.GRID_SEARCH)

            print('[INFO] Saving predictions from {}...'.format(feature_name))
            np.save(preds_filename, predictions_features)
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
    print('Execution duration: {:.2f} minutes {:.2f} seconds'.format(minutes, seconds), file=f)

## SEND TEXT

with open(MAIN_LOG) as f:
    text = f.read()
send(text)
