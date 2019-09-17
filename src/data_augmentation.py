import os
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt


IMG_SIZE = (224, 224, 3)


def run_data_augmentation():

    print("[INFO] Data augmentation...")

    # open image
    filename = '/datasets/devset/dev-set/frames/video5597-frame-48.jpg'
    image = plt.imread(filename)
    print('Image: {}, Shape: {}'.format(filename, image.shape))
    plt.imsave('aug_samples/image.png', image)
    paths = [
        filename,
    ]
    df = pd.DataFrame({ 'path': paths })

    # construct the image generator for data augmentation
    print("[INFO] Constructing image generators for data augmentation...")
    aug = ImageDataGenerator(
        rotation_range=30, 
        fill_mode="nearest")

    generator = aug.flow_from_dataframe(
        dataframe=df,
        x_col='path',
        class_mode=None,
        target_size=(1080, 1920),
        batch_size=1,
    )

    total = 0
    for images in generator:
        image = images[0]
        print(image.shape)
        plt.imsave('aug_samples/{}.png'.format(total), (image * 255).astype(np.uint8))
        total += 1
        if total == 10:
            break


if __name__ == "__main__":
    run_data_augmentation()
