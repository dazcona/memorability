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

## Deployment

1. Download the dataset (you may want to use an external drive) - https://stackoverflow.com/questions/113886/how-to-recursively-download-a-folder-via-ftp-on-linux
```
$ wget -m --ftp-user="<user>" --ftp-password=<password> ftp://<ftp server>
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

6. Run the code:
```
$ python src/workflow_cv.py
```

## Resources

* MediaEval 2018: http://multimediaeval.org/mediaeval2018/memorability/index.html
* MediaEval 2018: Predicting Media Memorability: https://www.slideshare.net/multimediaeval/mediaeval-2018-predicting-media-memorability
* Proceedings of the MediaEval 2018 Workshop: http://ceur-ws.org/Vol-2283/
* https://www.pyimagesearch.com/2019/01/21/regression-with-keras/
* https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
* https://towardsdatascience.com/machine-learning-word-embedding-sentiment-classification-using-keras-b83c28087456
* https://nlp.stanford.edu/projects/glove/
* https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py
