import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap


def visualize(X, y, y_pred, sample_num, figsize=(10, 10), cmap='viridis'):
    y_pred_np = y_pred[sample_num, :, :, :]
    y_class = np.argmax(y_pred_np, axis=-1)
    x = X.numpy()[sample_num, :, :, :]
    y_np = y.numpy()[sample_num, :, :, :]
    y_np = np.argmax(y_np, axis=-1)
    fig, axis = plt.subplots(1, 2, figsize=figsize)
    axis[0].imshow(x, cmap='gray')
    axis[0].imshow(y_np, cmap=cmap, alpha=0.3)
    axis[0].set_title("original labels")
    axis[1].imshow(x, cmap='gray')
    axis[1].imshow(y_class, cmap=cmap, alpha=0.3)
    axis[1].set_title("predicted labels")
    plt.show()


def visualize_non_empty_predictions(X, y, models, figsize=(10, 10), cmap=pl.cm.tab10_r, alpha=0.8, titles=[]):
    # create cmap
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, -1] = 0.9
    my_cmap[0, -1] = 0.1
    my_cmap = ListedColormap(my_cmap)

    n_plots = len(models) + 1
    fig, axis = plt.subplots(1, n_plots, figsize=figsize)

    axis[0].imshow(X, cmap='gray')
    # axis[0].imshow(y,cmap='gray')
    axis[0].set_title("original labels")
    axis[0].set_xticks([])
    axis[0].set_yticks([])

    for i, model in enumerate(models):
        y_pred = model.predict(tf.expand_dims(X, axis=0))
        y_pred[y_pred<=0] = 0
        y_pred[y_pred>0]  = 1
        # print(y_pred.shape)
        # print(y_pred)
        # print(y_class[0].shape)
        # axis[i+1].imshow(X, cmap='gray')
        axis[i+1].imshow(np.squeeze(y_pred,axis=0))
        if titles == []:
            axis[i+1].set_title(f"{model.name}")
        else:
            axis[i+1].set_title(f"{titles[i]}")
        axis[i+1].set_xticks([])
        axis[i+1].set_yticks([])

    plt.show()
