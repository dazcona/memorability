import os
import config
import numpy as np
from tqdm import tqdm


def get_aesthetics(videos, aesthetics_path):
    """
    Get our aesthetics features by reading each frame's numpy filename
    """

    print("[INFO] Extracting features from {:,} videos...".format(len(videos)))

    features = []

    for video in tqdm(videos):
        
        video_features = []

        for frame in config.FRAME_NUMBERS:

            # filename
            filename = os.path.join(aesthetics_path, video.split('.webm')[0] + '-frame-{}.npy'.format(frame))
            # features per frame
            frame_features = np.load(filename)

            video_features.append(frame_features)

        # features per video
        video_features = np.array(video_features).flatten()
        features.append(video_features)
    
    features = np.array(features)

    return features


def train_our_aesthetics(train_videos, val_videos, aesthetics_path):
    """ Get our aesthetics features for training and validation sets """

    X_train = get_aesthetics(train_videos, aesthetics_path)
    X_val = get_aesthetics(val_videos, aesthetics_path)

    return X_train, X_val


def videos_to_our_aesthetics(videos, aesthetics_path):
    """ Get our aesthetics features for a given set of videos """
    
    features = get_aesthetics(videos, aesthetics_path)

    return features
