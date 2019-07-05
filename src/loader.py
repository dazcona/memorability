# imports
import os
import numpy as np
import pandas as pd
import config


def data_load(FEATURES_DICT):
    """ Data loader """

    # Groudtruth
    print('[INFO] Loading groundtruth data...')
    dev_dataframe = pd.read_csv(config.DEV_GROUNDTRUTH) 

    # Include captions?
    if FEATURES_DICT['CAPTIONS']:

        print('[INFO] Loading captions data...')
        # Load captions
        dev_captions = pd.read_csv(config.DEV_CAPTIONS, sep='\t', header=None, names=['video', 'caption'] )
        # Fix captions
        dev_captions["caption"] = dev_captions["caption"].str.split('-').apply(' '.join)
        # Merge captions
        dev_dataframe = dev_dataframe.merge(dev_captions)

    # Pre-computed features?
    for feature_name, video_feature in zip(
        [ 'C3D', 'AESTHETICS', 'HMP' ], 
        [ FEATURES_DICT['C3D'], FEATURES_DICT['AESTHETICS'], FEATURES_DICT['HMP'] ]):

        if video_feature:
            print('[INFO] Loading {} video data...'.format(feature_name))
            # Load video feature data
            dataframe = get_video_features(feature_name)
            # Merge video feature data
            dev_dataframe = dev_dataframe.merge(dataframe)

    return dev_dataframe


def get_video_features(feature_name):
    """ Get video features """

    if feature_name == 'C3D':
        videos_path = config.DEV_C3D_FEATURE
        features_dim = config.C3D_DIM
    elif feature_name == 'AESTHETICS':
        videos_path = config.DEV_AESTHETIC_FEATURE
        features_dim = config.AESTHETIC_DIM
    elif feature_name == 'HMP':
        videos_path = config.DEV_HMP_FEATURE
        features_dim = config.HMP_DIM
    else:
        raise Exception('Feature {} is not recognized as a valid video feature'.format(feature_name))

    # Filenames with the features per video
    video_filenames = os.listdir(videos_path)

    # CSV features file
    csv_features_filename = 'data/{}_train.csv'.format(feature_name)
    if not os.path.isfile(csv_features_filename):
        # Create it
        print('[INFO] Creating {}...'.format(csv_features_filename))
        write_filename(csv_features_filename, videos_path, video_filenames, feature_name, features_dim)
    
    # Read features in CSV format
    dataframe = pd.read_csv(csv_features_filename)

    return dataframe


def write_filename(csv_filename, path, filenames, feature_name, features_dim):
    """ Write CSV file with all the features for each video """

    with open(csv_filename, 'w') as f: 
        # Header
        header = 'video,' + ','.join([ '{}_{}'.format(feature_name, i) for i in range(features_dim) ])
        f.write(header + '\n')
        # Rows
        for i, video in enumerate(filenames):
            # print('Iteration: {}'.format(i))
            # Get features per video
            video_features = video.split('.txt')[0] + '.webm'
            video_features += ',' + read_file_contents(os.path.join(path, video), feature_name)
            # write it!
            f.write(video_features + '\n')


# def get_contents(path, filenames):
#     """ Get contents video filenames """
#     # Collect features
#     features = []
#     for i, video in enumerate(filenames):
#         video_features = read_file_contents(os.path.join(path, video), feature_name)
#         video_features['video'] = video.split('.txt')[0] + '.webm'
#         features.append(video_features)
#     # Save as a dataframe
#     dataframe = pd.DataFrame(features)


def read_file_contents(filename, feature_name):
    """ Read the contents of the file for video processing features like C3D and AESTHETICS"""
    
    with open(filename) as f:
        if feature_name == 'C3D':
            return ','.join([ feature for feature in f.read().split() ])
        elif feature_name == 'AESTHETICS':
            return ','.join([ feature for feature in f.read().split(',') ])
        elif feature_name == 'HMP':
            return ','.join([ feature.split(':')[1] for feature in f.read().split() ])


# def read_file_contents(filename, feature_name):
#     """ Read the contents of the file for video processing features like C3D and Aesthetics"""
#     with open(filename) as f:
#         if feature_name == 'C3D':
#             return { '{}_{}'.format(feature_name, i + 1) : float(feature) for i, feature in enumerate(f.read().split()) }
#         elif feature_name == 'AESTHETICS':
#             return { '{}_{}'.format(feature_name, i + 1) : float(feature) for i, feature in enumerate(f.read().split(',')) }
#         elif feature_name == 'HMP':
#             return { '{}_{}'.format(feature_name, i + 1) : float(feature.split(':')[1]) for i, feature in enumerate(f.read().split()) }


def load_pretrained_word_vectors():
    """ Load pre-trained GLoVe vectors """

    print('[INFO] Loading word vectors...')
    embeddings_index = {}
    with open(config.GLOVE_FILE) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs
    print('[INFO] Found {:,} word vectors.'.format(len(embeddings_index)))
    return embeddings_index


if __name__ == "__main__":
    FEATURES_DICT = {
        'CAPTIONS': True,
        'C3D': True,
        'AESTHETICS': True,
        'HMP': True,
    }
    # Loading Groudtruth
    dataframe = data_load()
    print(dataframe.head())
    # Loading Groudtruth + Captions
    dataframe = data_load(FEATURES_DICT)
    print(dataframe.head())