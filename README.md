# Insight@DCU in the Memorability Challenge at MediaEval2019

This task focuses on the problem of predicting how memorable a video is to viewers. It requires participants to automatically predict memorability scores for videos that reflect the probability a video will be remembered. 

Task participants are provided with an extensive dataset of videos that are accompanied by memorability annotations, as well as pre-extracted state-of-the-art visual features. The ground truth has been collected through recognition tests, and thus results from objective measurement of memory performance. Participants will be required to train computational models capable of inferring video memorability from visual content. Optionally, descriptive titles attached to the videos may be used. Models will be evaluated through standard evaluation metrics used in ranking tasks (Spearman’s rank correlation). The data set used in 2019, is the same as in 2018 (2018’s testset ground truth data has not been released). This year the task focuses on understanding the patterns in the data and improving the ability of algorithms to capture those patterns.

## Challenge

http://www.multimediaeval.org/mediaeval2019/memorability/

## Dataset

The dataset is composed of 10,000 (soundless) short videos extracted from raw footage used by professionals when creating content, and in particular, commercials. Each video consists of a coherent unit in terms of meaning and is associated with two scores of memorability that refer to its probability to be remembered after two different durations of memory retention. 

These videos come with a set of pre-extracted features, such as: Dense SIFT, HoG descriptors, LBP, GIST, Color Histogram, MFCC, Fc7 layer from AlexNet, C3D features, etc.

## Technologies

* [Python 3](https://www.python.org/)
* [numpy](http://www.numpy.org)
* [pandas](https://pandas.pydata.org/)
* [matplotlib](https://matplotlib.org/)
* [scikit-learn](https://scikit-learn.org/)
* [pillow](https://pillow.readthedocs.io/)
* [keras](https://keras.io)
* [tensorflow](https://www.tensorflow.org/)
* [jupyter](https://jupyter.org/)
* [Docker](https://www.docker.com/)

## Results

See the following [table](https://docs.google.com/spreadsheets/d/1LrenTHNGRZzCHYluYl2YPdmlHuUsTJSMhUIb1X91hdk/)

## Deployment

1. Download the dataset (you may want to use an external drive) via FTP like [here](https://stackoverflow.com/questions/113886/how-to-recursively-download-a-folder-via-ftp-on-linux):
```
$ wget -m --ftp-user="<user>" --ftp-password=<password> ftp://<ftp server>
```
and unzip multi-part files like [here](http://koenaerts.ca/unzip-multi-part-archives-in-linux/):
```
$ zip --fix me18me-devset.zip --output mybigzipfile.zip
$ unzip mybigzipfile.zip
```

2. Mount the dataset as drive in */datasets* in *docker-compose.yml*. As an example:
```
volumes:
  - /Volumes/HDD/datasets/:/datasets
```

3. Build the docker image:
```
$ cd docker
$ make build
```

4. Create a docker container based on the image:
```
$ make run
```

5. SSH to the docker container:
```
$ make dev
```

6. Extract frames from videos:
```
$ python src/extract_frames.py
```

7. Run the training:
```
$ python src/workflow_split.py
```

## Activation Maps

Model: 'ResNet152' for frame 48 of the videos

### Top short-term most memorable videos

1. video798.webm

Predicted: [('n04456115', 'torch', 0.23151287), ('n03498962', 'hatchet', 0.094463184), ('n03141823', 'crutch', 0.0654099)]

2. video1981.webm

Predicted: [('n02883205', 'bow_tie', 0.99436283), ('n04456115', 'torch', 0.0010983162), ('n04418357', 'theater_curtain', 0.00067173946)]

3. video4903.webm

Predicted: [('n04404412', 'television', 0.5428618), ('n03180011', 'desktop_computer', 0.115691125), ('n04152593', 'screen', 0.11060062)]

4. video9496.webm

Predicted: [('n09421951', 'sandbar', 0.55648345), ('n09428293', 'seashore', 0.13317421), ('n09332890', 'lakeside', 0.03515112)]

5. video6103.webm

Predicted: [('n03404251', 'fur_coat', 0.66497004), ('n03045698', 'cloak', 0.16292651), ('n04229816', 'ski_mask', 0.024773473)]

### Top long-term most memorable videos

1. video5186.webm

Predicted: [('n03792782', 'mountain_bike', 0.8176742), ('n02835271', 'bicycle-built-for-two', 0.1651485), ('n04509417', 'unicycle', 0.009558631)]

2. video4798.webm

Predicted: [('n03594734', 'jean', 0.64808583), ('n02977058', 'cash_machine', 0.06661992), ('n04479046', 'trench_coat', 0.026500706)]

3. video480.webm

Predicted: [('n02097130', 'giant_schnauzer', 0.28221375), ('n02102318', 'cocker_spaniel', 0.172711), ('n02097298', 'Scotch_terrier', 0.11454323)]

4. video7606.webm

Predicted: [('n03000684', 'chain_saw', 0.15715672), ('n03976657', 'pole', 0.099422), ('n03532672', 'hook', 0.064023055)]

5. video4809.webm

Predicted: [('n04039381', 'racket', 0.9964013), ('n04409515', 'tennis_ball', 0.0032226138), ('n03942813', 'ping-pong_ball', 0.00037128705)]

## Resources

* MediaEval 2018: http://multimediaeval.org/mediaeval2018/memorability/index.html
* MediaEval 2018: Predicting Media Memorability: https://www.slideshare.net/multimediaeval/mediaeval-2018-predicting-media-memorability
* Proceedings of the MediaEval 2018 Workshop: http://ceur-ws.org/Vol-2283/
* Keras & Regression: https://www.pyimagesearch.com/2019/01/21/regression-with-keras/
* GloVe: https://nlp.stanford.edu/projects/glove/
* Pre-trained word embeddings: https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py
* https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
* https://towardsdatascience.com/machine-learning-word-embedding-sentiment-classification-using-keras-b83c28087456
* Keras custom metrics: https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/
* How to Train a Final Machine Learning Model: https://machinelearningmastery.com/train-final-machine-learning-model/
