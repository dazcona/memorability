import os
import numpy as np
import config
from loader import data_load
from sklearn.model_selection import train_test_split
from itertools import combinations_with_replacement
from evaluate import evaluate_spearman

## SEED

np.random.seed(42)


if __name__ == "__main__":
    
    ## LOAD DATA

    target = input("[INPUT] Which target? short[1] / long[otherwise] ")
    target = 'short-term_memorability' if target.strip() == '1' else 'long-term_memorability'
    print("[INFO] You selected: {}".format(target))

    print('[INFO] Loading data...')
    dataframe = data_load({
        'C3D': 0,
        'AESTHETICS': 0,
        'HMP': 0,
        'ColorHistogram': 0,
        'LBP': 0,
        'InceptionV3': 0,
        'CAPTIONS': 0,
        'PRE-TRAINED NN': 0,
        'FINE-TUNED NN': 0,
        'Emotions': 0,
    }, config.DEV_GROUNDTRUTH, config.DEV_CAPTIONS, config.DEV_FEATURES_PATH, 'train_val')

    X = dataframe.drop(columns=[ 'short-term_memorability', 'nb_short-term_annotations', 'long-term_memorability', 'nb_long-term_annotations' ])
    Y = dataframe[target]

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

    filenames = [ filename for filename in os.listdir('predictions/training/') if filename.startswith(target) ]
    print('Filenames:')
    print('\n'.join(filenames))

    # get weights
    weights = {}
    for filename in filenames:
        
        weight = input("[INPUT] What weight do you want for file: {}? Break[b] ".format(filename)).strip()
        if weight == '':
            continue
        if weight.lower() == 'b':
            break
        weights[filename] = float(weight)

    print("[INFO] Weights: {}".format(weights))

    assert round(sum(weights.values())) == 1, 'ERROR: Sum of prediction weights must be equal to 1'
        
    # apply weights to predictions
    predictions = []
    for filename, weight in weights.items():
        if weight > 0:
            print("[INFO] Reading: {} with weight: {}".format(filename, weight))
            predictions.append(
                np.load(os.path.join(config.PREDICTIONS_TRAIN, filename)) * weight
            )
    predictions = np.sum(predictions, axis=0)

    # evaluate
    corr_coefficient, p_value = evaluate_spearman(y_val, predictions)
    print('[INFO] Spearman Correlation Coefficient: {:.5f} (p-value {:.5f})'.format(corr_coefficient, p_value))
