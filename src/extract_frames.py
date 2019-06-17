import os
import sys
import config
import pandas as pd
import cv2

# Data
df_groundtruth = pd.read_excel(config.GROUNDTRUTH)
groundtruth = df_groundtruth.to_dict('records')

print(df_groundtruth.columns)

# config.SOURCES

for movie in groundtruth:
    
    # Filename
    sequence_name = movie['sequence_name']
    filename = os.path.join(config.SOURCES, sequence_name + '.mp4')

    if not os.path.isfile(filename):
        print('File {} does not exist'.format(filename))
        continue

    print('Reading file {}'.format(filename))

    # Read Video!
    success = True
    video = cv2.VideoCapture(filename)
    success, image = video.read()

    # Frame
    frame_id = int(video.get(1))
    cv2.imwrite("frames/{}-frame-{}.jpg".format(sequence_name, frame_id), image)
    
    # Frames per second
    fps = int(video.get(cv2.CAP_PROP_FPS))
    
    # Collect 1 frame every 1 second
    seconds = 1
    multiplier = fps * seconds

    while success:

        # Frame
        frame_id = int(video.get(1))

        if frame_id % multiplier == 0:
            # Save it
            cv2.imwrite("frames/{}-frame-{}.jpg".format(sequence_name, frame_id), image)

        # Read Video!
        success, image = video.read()

    video.release()

    # exit
    sys.exit(2)

