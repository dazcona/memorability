from keras.applications import ResNet152
import numpy as np

# frame numbers
FRAME_NUMBERS = [1, 24, 48, 72, 96, 120, 144, 168]
IMG_SIZE = (224, 224, 3)
NEURONS = 256
LEARNING_RATE = 1e-3
EPOCHS = 200
DECAY = 1e-3 / EPOCHS

def train_fine_tuned_cnn(train_videos, val_videos, y_train, y_val, frames_path):
    
    print("[INFO] Fine-tuning CNN...")

    # train images
    train_image_paths = [ 
        os.path.join(frames_path, video.split('.webm')[0] + '-frame-{}.jpg'.format(frame))
        for video in train_videos
        for frame in FRAME_NUMBERS
    ]
    print("[INFO] Training images: {:,}...".format(len(train_image_paths)))

    # validation images
    val_image_paths = [ 
        os.path.join(frames_path, video.split('.webm')[0] + '-frame-{}.jpg'.format(frame))
        for video in val_videos
        for frame in FRAME_NUMBERS
    ]
    print("[INFO] Validation images: {:,}...".format(len(val_image_paths)))

    print("[INFO] Collecting images before data augmentation...")
    train_img = []
    for image_path in train_image_paths:
        image = load_img(image_path, target_size=IMG_SIZE)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        # image = imagenet_utils.preprocess_input(image)
        train_img.append(image)

    val_img = []
    for image_path in val_image_paths:
        image = load_img(image_path, target_size=IMG_SIZE)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        val_img.append(image)

    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest")
    
    # Base Model
    print("[INFO] Constructing model...")
    base_model = ResNet152(weights='imagenet', include_top=False, input_tensor=Input(shape=IMG_SIZE))

    # Head Model
    head_model = base_model.output
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(NEURONS, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(classes, activation="softmax")(head_model)

    # Combine models
    model = Model(inputs=base_model.input, outputs=head_model)

    print("[INFO] Freezing the model's layers...")
    # loop over all layers in the base model and freeze them so they
    # will *not* be updated during the training process
    for layer in base_model.layers:
        layer.trainable = False

    # compile our model (this needs to be done after our setting our
    # layers to being non-trainable
    print("[INFO] Compiling model...")
    # Optimizer
    opt = Adam(lr=LEARNING_RATE, decay=DECAY)
    model.compile(
        loss='mean_squared_error',
        optimizer=opt,
        metrics=['mse', 'mae', 'mape'],
    )

    # train the head of the network for a few epochs (all other
    # layers are frozen) -- this will allow the new FC layers to
    # start to become initialized with actual "learned" values
    # versus pure random
    print("[INFO] Training model's head...")
    model.fit_generator(aug.flow(train_img, y_train, batch_size=32),
        validation_data=(val_img, y_val), epochs=EPOCHS,
        steps_per_epoch=len(train_img) // 32, verbose=1)

    # evaluate the network after initialization
    print("[INFO] Predicting values...")
    predictions = model.predict(val_img, batch_size=32)

    # Evaluate
    # print(classification_report(testY.argmax(axis=1),
    # predictions.argmax(axis=1), target_names=classNames))

    # now that the head FC layers have been trained/initialized, lets
    # unfreeze the final set of CONV layers and make them trainable
    # for layer in base_model.layers[15:]:
    #     layer.trainable = True

    # ...

