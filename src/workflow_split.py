## VALUES

TARGET = 'short-term_memorability'
TARGET_COLS = [ 'short-term_memorability', 'nb_short-term_annotations', 'long-term_memorability', 'nb_long-term_annotations' ]
CAPTIONS = True
C3D = False
AESTHETICS = False
HMP = False
ALGORITHM = 'SVM'

## LOAD DATA

from loader import data_load

print('[INFO] Loading data with captions...')
dataframe = data_load(captions=CAPTIONS, C3D=C3D, Aesthetics=AESTHETICS, HMP=HMP)

X = dataframe.drop(columns=TARGET_COLS)
Y = dataframe[TARGET]

## SPLIT DATA

from sklearn.model_selection import train_test_split

print('[INFO] Splitting data between train (80%) and validation (20%) sets...')
X_train, X_val, y_train, y_val = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42)

## CAPTIONS

from tfidf import fit_and_transform_text

print('[INFO] Processing the captions and transforming them into numbers...')
X_train_tfidf, X_val_tfidf = fit_and_transform_text(X_train['caption'], X_val['caption'])

## MODELLING

from modelling import train_predict

predictions = train_predict(X_train_tfidf, y_train, X_val_tfidf, method=ALGORITHM)

## EVALUATE

from evaluate import evaluate_spearman

print('[INFO] Evaluating the performance of the predictions...')
corr_coefficient, p_value = evaluate_spearman(y_val, predictions)
print('[INFO] Spearman Correlation Coefficient: {:.4f} (p-value {:.4f})'.format(corr_coefficient, p_value))

## PLOT EVALUATION

import matplotlib.pyplot as plt
font = { 'family': 'DejaVu Sans', 'weight': 'bold', 'size': 15 }
plt.rc('font', **font)

print('[INFO] Plotting true values vs predicted values...')
plt.figure(figsize=(12, 10), dpi=100)
plt.scatter(y_val, predictions, c="b", alpha=0.25)
plt.title("True values vs Predicted values")
plt.xlabel("True values")
plt.ylabel("Predicted valuesy")
plt.savefig('figures/true_vs_predicted.png')
