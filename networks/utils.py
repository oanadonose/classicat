import datetime
from datetime import datetime
import matplotlib.pyplot as plt

import sklearn
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import os
import csv
import numpy as np
import itertools

def testnp2():
    return np.random.randint(20)

def testnp():
    return np.random.randint(20)

def curTime():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    return now

def plotLoss(title, history):
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, 
                       axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.rc('font', size=10)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def displayCM(title, y_true, predictions, plot_labels):
    y_pred = np.argmax(predictions, axis=-1)
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm=cm, classes=plot_labels, title=title)
    
## directory to csv
def create_csv(filename, path):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id','label'])
        for breed in os.listdir(path):
            files=os.listdir(path+'/'+breed)
            for file in files:
                writer.writerow([file, breed])