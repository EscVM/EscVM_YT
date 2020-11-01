# Lint as: python
#
# Authors: Vittorio 
# Location: Turin, Biella
#

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def plotImages(images_batch, img_n, classes):
    """
    Take as input a batch from the generator and plt a number of images equal to img_n
    Default columns equal to max_c. At least inputs of batch equal two
    """
    max_c = 5
    
    if img_n <= max_c:
        r = 1
        c = img_n
    else:
        r = math.ceil(img_n/max_c)
        c = max_c
        
    fig, axes = plt.subplots(r, c, figsize=(15,15))
    axes = axes.flatten()
    for img_batch, label_batch, ax in zip(images_batch[0], images_batch[1], axes):
        ax.imshow(img_batch)
        ax.grid()
        ax.set_title('Class: {}'.format(classes[label_batch]))
    plt.tight_layout()
    plt.show()

def plotHistory(history):
    """
    Plot the loss and accuracy curves for training and validation 
    """
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1.5)
    plt.show()


def explainGradCam(explainer, ax, img, y, model_1, y_pred_1, model_2, y_pred_2, class_names):
    """
    [Attention Episode]
    Plot GRADCAM of two trained models. It needs an axes with two columns
    """
    data = ([img], None)
    
    y_predm_1 = np.argmax(y_pred_1)
    y_predm_2 = np.argmax(y_pred_2)
        
    grid_1 = explainer.explain(data, model_1, class_index=y_predm_1, image_weight=0.8)
    grid_2 = explainer.explain(data, model_2, class_index=y_predm_2, image_weight=0.8)

    ax[0].set_xlabel("Pred: {} {:2.0f}% ({})".format(class_names[y_predm_1],
                                100*np.max(y_pred_1),
                                class_names[y]),
                                color=('blue' if y == y_predm_1 else 'red'))

    
    ax[1].set_xlabel("Pred: {} {:2.0f}% ({})".format(class_names[y_predm_2],
                                100*np.max(y_pred_2),
                                class_names[y]),
                                color=('blue' if y == y_predm_2 else 'red'))
    ax[0].imshow(grid_1)
    ax[1].imshow(grid_2)