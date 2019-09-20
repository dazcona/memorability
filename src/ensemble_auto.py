import os
import numpy as np
from loader import data_load
from sklearn.model_selection import train_test_split
from itertools import combinations_with_replacement
from evaluate import evaluate_spearman
import csv
# import pdb; pdb.set_trace()


## SEED

np.random.seed(42)


if __name__ == "__main__":

    ## LOAD DATA

    target = 'long-term_memorability'
    print("[INFO] Target: {}".format(target))

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
    })

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

    yoyo = [
        'short-term_memorability_LBP.npy',
        'short-term_memorability_C3D.npy',
        'short-term_memorability_HMP.npy',
        'short-term_memorability_captions_embeddings.npy',
        'short-term_memorability_pretrained_ResNet152.npy', 
        'short-term_memorability_our_aesthetics_BR.npy',
        'short-term_memorability_emotions_ML.npy',
    ]
    # Removed: 'short-term_memorability_pretrained_ResNet50.npy', 'short-term_memorability_AESTHETICS.npy',
    my_filenames = [
        'long-term_memorability_HMP.npy',
        'long-term_memorability_C3D.npy',
        'long-term_memorability_LBP.npy',
        'long-term_memorability_captions_embeddings.npy',
        'long-term_memorability_pretrained_ResNet152.npy',
        'long-term_memorability_our_aesthetics_BR.npy',
        'long-term_memorability_emotions_ML.npy',
    ]
    # Removed: 'long-term_memorability_pretrained_ResNet50.npy', 'long-term_memorability_fine_tuned.npy'
    # 'long-term_memorability_AESTHETICS.npy', 'long-term_memorability_ColorHistogram.npy',
    N_FILES = 20

    results = {}

    comb = combinations_with_replacement(my_filenames, N_FILES)
    for i, combination in enumerate(comb):

        print('{}. {}'.format(i, combination))

        predictions = []
        text = []

        for my_filename in my_filenames:

            times = combination.count(my_filename)
            if times > 0:

                weight = times * 1. / N_FILES
                predictions.append(
                    np.load(os.path.join('predictions', my_filename)) * weight
                )

                short_name = my_filename.split(target + '_')[1].split('.npy')[0].replace('pretrained_', '').replace('_embeddings', '').title()
                text.append('{} ({:.3f})'.format(short_name, weight))

        predictions = np.sum(predictions, axis=0)

        # evaluate
        corr_coefficient, p_value = evaluate_spearman(y_val, predictions)
        print('[INFO] Spearman Correlation Coefficient: {:.5f} (p-value {:.5f})'.format(corr_coefficient, p_value))    

        # save
        results[corr_coefficient] = (', '.join(text), p_value)

    with open('reports/auto_ensemble_{}.csv'.format(target), 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        filewriter.writerow(['Score (Spearman Correlation Coefficient)', 'p-value', 'Weights'])
        for k, v in sorted(results.items(), reverse=True):
            filewriter.writerow([k, v[1], v[0]])
            # print('Score: {:.5f}, Weights: {}'.format(k, v))
