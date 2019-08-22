# imports
import os
import numpy as np
import config
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input, GRU, Dense
from keras.initializers import Constant
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt
# import keras.backend as K


def train_embeddings_network(train_captions, y_train, validation_captions, y_val, embeddings_index, fold):

    # CAPTIONS

    train_captions = train_captions.tolist()
    caption_words = list(set([ word for caption in train_captions for word in caption.split() ]))
    print('[INFO] {:,} words in the dev captions'.format(len(caption_words)))

    # VECTORIZATION

    print('[INFO] Vectorize the captions into a 2D integer tensor...')
    # Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_captions)
    # Sequences
    train_sequences = tokenizer.texts_to_sequences(train_captions)
    # Padding
    word_index = tokenizer.word_index
    MAX_NUM_WORDS = len(word_index)
    MAX_SEQUENCE_LENGTH = max([ len(caption.split()) for caption in train_captions ])
    
    # X and Y TRAIN
    X_train = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    y_train = np.array(y_train)

    print('[INFO] Creating the embeddings matrix...')
    # prepare embedding matrix
    num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
    embedding_matrix = np.zeros((num_words, config.EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    print('[INFO] Embedding Matrix\'s shape is {}'.format(embedding_matrix.shape))

    # MODEL

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                config.EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    print('[INFO] Training GRU model...')
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = GRU(units=32, dropout=0.5, recurrent_dropout=0.5)(embedded_sequences)
    preds = Dense(1, activation='sigmoid')(x)
    model = Model(sequence_input, preds)

    print('[INFO] Model\'s Summary')
    print(model.summary())

    # COMPILE

    print('[INFO] Compiling model...')
    
    # Optimizer
    opt = Adam(lr=1e-3, decay=1e-3 / 200)

    model.compile(
        loss='mean_squared_error',
        optimizer=opt,
        metrics=['mse', 'mae', 'mape', 'cosine'],
    )

    # X and Y TEST
    print('[INFO] Preprocessing validation captions...')
    validation_captions = validation_captions.tolist()
    validation_sequences = tokenizer.texts_to_sequences(validation_captions)
    X_val = pad_sequences(validation_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    y_val = np.array(y_val)

    # FIT

    print('[INFO] Fitting model...')
    tensorboard = TensorBoard(log_dir=config.RUN_LOG_FOLD_DIR.format(fold)) # config.RUN_LOG_DIR
    #checkpoints = ModelCheckpoint(
    #    os.path.join(
    #        config.RUN_CHECKPOINT_DIR,
    #        'weights.fold_' + str(fold) + '.{epoch:02d}-{val_loss:.2f}.hdf5'),
    #    monitor='mean_squared_error', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    H = model.fit(X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.NUM_EPOCHS, 
        shuffle=False,
        callbacks=[
            tensorboard,
            # checkpoints,
            ]
    )

    # PLOT TRAINING LOSS vs ACCURACY

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, config.NUM_EPOCHS), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, config.NUM_EPOCHS), H.history["val_loss"], label="val_loss")
    # plt.plot(np.arange(0, config.NUM_EPOCHS), H.history["mean_squared_error"], label="train_MSE")
    # plt.plot(np.arange(0, config.NUM_EPOCHS), H.history["val_mean_squared_error"], label="val_MSE")
    plt.plot(np.arange(0, config.NUM_EPOCHS), H.history["mean_absolute_error"], label="train_MAE")
    plt.plot(np.arange(0, config.NUM_EPOCHS), H.history["val_mean_absolute_error"], label="val_MAE")
    # plt.plot(np.arange(0, config.NUM_EPOCHS), H.history["mean_absolute_percentage_error"], label="train_MAPE")
    # plt.plot(np.arange(0, config.NUM_EPOCHS), H.history["val_mean_absolute_percentage_error"], label="val_MAPE")
    # plt.plot(np.arange(0, config.NUM_EPOCHS), H.history["cosine_proximity"], label="train_cosine_prox")
    # plt.plot(np.arange(0, config.NUM_EPOCHS), H.history["val_cosine_proximity"], label="val_cosine_prox")
    plt.title("Training Loss and MSE")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/MSE")
    plt.legend()
    plt.savefig('{}/embeddings_loss_vs_MSE.png'.format(config.RUN_LOG_FOLD_DIR.format(fold)))

    print('[INFO] Predicting values...')
    predicted = model.predict(X_val)

    ## PLOT EVALUATION

    print('[INFO] Plotting true values vs predicted values...')
    font = { 'family': 'DejaVu Sans', 'weight': 'bold', 'size': 15 }
    plt.rc('font', **font)
    plt.figure(figsize=(12, 10), dpi=100)
    plt.scatter(y_val, predicted, c="b", alpha=0.25)
    plt.title("True values vs Predicted values")
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.savefig('{}/true_vs_predicted.png'.format(config.RUN_LOG_FOLD_DIR.format(fold)))

    # return predictions

    return predicted
    
