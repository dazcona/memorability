# imports
import numpy as np
import csv
import pandas as pd
import config

## SUBMISSIONS

SUBMISSIONS = {
    'shortterm': {
        1: { 'Captions': 0.40, 'Resnet152': 0.55, 'Emotions': 0.05, 'C3D': 0., 'LBP': 0., 'Aesthetics': 0. },
        2: { 'Captions': 0.40, 'Resnet152': 0.55, 'Emotions': 0.05, 'C3D': 0., 'LBP': 0., 'Aesthetics': 0. },
        3: { 'Captions': 0.40, 'Resnet152': 0.55, 'Emotions': 0.05, 'C3D': 0., 'LBP': 0., 'Aesthetics': 0. },
        4: { 'Captions': 0.40, 'Resnet152': 0.55, 'Emotions': 0.05, 'C3D': 0., 'LBP': 0., 'Aesthetics': 0. },
        5: { 'Captions': 0.40, 'Resnet152': 0.55, 'Emotions': 0.05, 'C3D': 0., 'LBP': 0., 'Aesthetics': 0. },
    },
    'longterm': {
        5: { 'Captions': 0.40, 'Resnet152': 0.55, 'Emotions': 0.05, 'C3D': 0., 'LBP': 0., 'Aesthetics': 0. },
        1: { 'Captions': 0.40, 'Resnet152': 0.55, 'Emotions': 0.05, 'C3D': 0., 'LBP': 0., 'Aesthetics': 0. },
        2: { 'Captions': 0.40, 'Resnet152': 0.55, 'Emotions': 0.05, 'C3D': 0., 'LBP': 0., 'Aesthetics': 0. },
        3: { 'Captions': 0.40, 'Resnet152': 0.55, 'Emotions': 0.05, 'C3D': 0., 'LBP': 0., 'Aesthetics': 0. },
        4: { 'Captions': 0.40, 'Resnet152': 0.55, 'Emotions': 0.05, 'C3D': 0., 'LBP': 0., 'Aesthetics': 0. },
    }
}
SUBMISSION_NAME = 'runs/me19mem_insightdcu_{}_run{}.csv'

## PREDICTIONS FILENAMES
PREDICTIONS = {
    'shortterm': {
        'Captions': 'predictions/test/short-term_memorability_captions_embeddings.npy', 
        'Resnet152': 'predictions/test/short-term_memorability_pretrained_ResNet152.npy', 
        'Emotions': 'predictions/test/short-term_memorability_emotions.npy', 
        'C3D': 'predictions/test/short-term_memorability_C3D.npy', 
        'LBP': 'predictions/test/short-term_memorability_LBP.npy', 
        'Aesthetics': 'predictions/test/short-term_memorability_our_aesthetics.npy',
    },
    'longterm': {
        'Captions': 'predictions/test/long-term_memorability_captions_embeddings.npy', 
        'Resnet152': 'predictions/test/long-term_memorability_pretrained_ResNet152.npy', 
        'Emotions': 'predictions/test/long-term_memorability_emotions.npy', 
        'C3D': 'predictions/test/long-term_memorability_C3D.npy', 
        'LBP': 'predictions/test/long-term_memorability_LBP.npy', 
        'Aesthetics': 'predictions/test/long-term_memorability_our_aesthetics.npy',
    }
}

# Test videos
dataframe = pd.read_csv(config.TEST_GROUNDTRUTH, sep="\n", header=None, names=['video'])
videos = list(dataframe['video'].str.replace('.txt', '.webm').str.strip())
print('[INFO] Number of test samples: {:,}'.format(len(videos)))

for target in SUBMISSIONS.keys():

    for run_number, weights in SUBMISSIONS[target].items():

        # submission name
        submission = SUBMISSION_NAME.format(target, run_number)
        print('[INFO] Preparing {}...'.format(submission))

        predictions = []

        for name, weight in weights.items():
            
            if weight > 0:
                print("[INFO] Reading: {} with weight: {}".format(name, weight))
                filename = PREDICTIONS[target][name]
                predictions.append(
                    np.load(filename) * weight
                )

        predictions = np.sum(predictions, axis=0)

        # write them!
        with open(submission, 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',')
            # filewriter.writerow(['video name', 'memorability score', 'confidence value'])
            for video, prediction in zip(videos, predictions):
                filewriter.writerow( [video, prediction, 1] )
