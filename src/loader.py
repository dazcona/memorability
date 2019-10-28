# imports
import os
import numpy as np
import pandas as pd
import config
# import pdb; pdb.set_trace()


def data_load(FEATURES_WEIGHTS, GROUNDTRUTH, CAPTIONS, FEATURES_PATH, dev_or_test):
    """ Data loader """

    # Groudtruth
    print('[INFO] Loading groundtruth data...')
    if 'train' in dev_or_test: 
        dataframe = pd.read_csv(GROUNDTRUTH)
    else: # test
        dataframe = pd.read_csv(GROUNDTRUTH, sep="\n", header=None, names=['video'])
        dataframe['video'] = dataframe['video'].str.replace('.txt', '.webm').str.strip()

    # Include captions?
    if FEATURES_WEIGHTS['CAPTIONS'] > 0:

        print('[INFO] Loading captions data...')
        # Load captions
        df_captions = pd.read_csv(CAPTIONS, sep='\t', header=None, names=['video', 'caption'] )
        # Fix captions
        df_captions["caption"] = df_captions["caption"].str.split('-').apply(' '.join)

        # Merge captions
        dataframe = dataframe.merge(df_captions)

    # Pre-computed features?
    for feature_name in [ 'C3D', 'AESTHETICS', 'HMP', 'ColorHistogram', 'LBP', 'InceptionV3' ]:

        if FEATURES_WEIGHTS[feature_name] > 0:

            print('[INFO] Loading {} video data...'.format(feature_name))
            # Load video feature data
            video_dataframe = get_video_features(feature_name, FEATURES_PATH, dev_or_test)

            # Merge video feature data
            dataframe = dataframe.merge(video_dataframe)   

    return dataframe


def get_video_features(feature_name, features_path_dict, dev_or_test):
    """ Get video features """

    if feature_name not in features_path_dict:
        raise Exception('Feature {} is not recognized as a valid video feature'.format(feature_name))
    videos_path = features_path_dict[feature_name]
    features_dim = config.FEATURES_DIM[feature_name]

    # Filenames with the features per video
    video_filenames = os.listdir(videos_path)

    # CSV features file
    csv_features_filename = '{}/{}_{}.csv'.format(config.MY_FEATURES_DIR, feature_name, dev_or_test)
    if not os.path.isfile(csv_features_filename):
        # Create it
        print('[INFO] Creating {}...'.format(csv_features_filename))
        write_filename(csv_features_filename, videos_path, video_filenames, feature_name, features_dim)

    # Read features in CSV format
    dataframe = pd.read_csv(csv_features_filename)

    # Fill Nan
    dataframe.fillna(0, inplace=True)

    return dataframe


def write_filename(csv_filename, path, filenames, feature_name, features_dim):
    """ Write CSV file with all the features for each video """

    if feature_name in ['C3D', 'AESTHETICS']:

        # Fixed number of columns per video file

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

    elif feature_name in ['HMP']:

        # Variable number of columns per video file

        # save as array of dictionaries
        videos = []
        for video in filenames:
            video_name = video.split('.txt')[0] + '.webm'
            # Get features per video
            v = read_file_contents(os.path.join(path, video), feature_name)
            v['video'] = video_name
            videos.append(v)

        # save as dataframe
        temp_dataframe = pd.DataFrame(videos)

        # save as csv
        temp_dataframe.to_csv(csv_filename, index=False)

    elif feature_name in ['ColorHistogram', 'InceptionV3']:

        # save as array of dictionaries
        videos = []
        video_names = list(set([ filename.split('-')[0] for filename in filenames ]))
        for video in video_names:
            video_name = video.split('.txt')[0] + '.webm'
            # Get features per video and per frame
            v = {}
            for frame_number in config.THREE_FRAMES:
                frame_file = '{}-{}.txt'.format(video.split('.txt')[0], frame_number)
                v.update(read_file_contents(os.path.join(path, frame_file), '{}_{}'.format(feature_name, frame_number)))
            v['video'] = video_name
            videos.append(v)

        # save as dataframe
        temp_dataframe = pd.DataFrame(videos)

        # save as csv
        temp_dataframe.to_csv(csv_filename, index=False)

    elif feature_name in ['LBP']:

        # Fixed number of columns per frame file

        with open(csv_filename, 'w') as f:
            # Header
            header = 'video,' + ','.join([ '{}_{}'.format(feature_name, i) for i in range(features_dim) ])
            f.write(header + '\n')
            # Rows
            videos = []
            video_names = list(set([ filename.split('-')[0] for filename in filenames ]))
            for video in video_names:
                video_features = video + '.webm'
                # Get features per video and per frame
                for frame_number in config.THREE_FRAMES:
                    video_features += ',' + read_file_contents(os.path.join(path, '{}-{}.txt'.format(video, frame_number)), feature_name)
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
        if feature_name == 'C3D' or feature_name == 'LBP':
            return ','.join([ feature for feature in f.read().split() ])
        elif feature_name == 'AESTHETICS':
            return ','.join([ feature for feature in f.read().split(',') ])
        elif feature_name == 'HMP' or feature_name.startswith('ColorHistogram')  or feature_name.startswith('InceptionV3'):
            return { '{}_{}'.format(feature_name, feature.split(':')[0]) : float(feature.split(':')[1]) 
                    for feature in f.read().split() }


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
    FEATURES_WEIGHTS = {
        'CAPTIONS': 0,
        'C3D': 0,
        'AESTHETICS': 0,
        'HMP': 0,
        'ColorHistogram': 0,
        'LBP': 0,
        'InceptionV3': 1,
    }
    # Loading Groudtruth + Captions
    dataframe = data_load(FEATURES_WEIGHTS, config.DEV_GROUNDTRUTH, 
                            config.DEV_CAPTIONS, config.DEV_FEATURES_PATH, 'train_val')
    print(dataframe.head())
