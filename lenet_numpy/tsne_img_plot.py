# That's an impressive list of imports.
import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

# We import sklearn.
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)
from sklearn.utils.extmath import _ravel
# Random state.
RS = 20150101

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
#%matplotlib inline

# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

import os
#from main import *

def combine_channels(ims, titles, nrows, ncols, name, digit):
    plt.figure(figsize=(8,8))
    plt.gray()
    for i in range(ncols * nrows):
        ax = plt.subplot(nrows, ncols, i + 1)
        ax.matshow(ims[i])
        plt.xticks([]); plt.yticks([])
        plt.title(titles[i])
    plt.savefig("combined_feature_maps/" + str(digit) + "/" + name + ".jpeg", dpi=150)

def get_images(name, num, batch_string):
    ims = []
    titles = []
    for i in range(num):
        ims += [plt.imread(name + str(i+1) + batch_string + ".jpeg")]
        t = 'Channel-' + str(i+1)
        titles += [t]
    return ims,  titles

def merge_images(digit, batch_string):
    nrows, ncols = 2, 3
    ims, tit = get_images("feature_maps/" + str(digit) + "/conv1_c", 6, batch_string)
    combine_channels(ims, tit, nrows, ncols, "conv1" + batch_string, digit)

    nrows, ncols = 4, 4
    ims, tit = get_images("feature_maps/" + str(digit) +  "/conv2_c", 16, batch_string)
    combine_channels(ims, tit, nrows, ncols, "conv2" + batch_string, digit)

    nrows, ncols = 2, 3
    ims, tit = get_images("feature_maps/" + str(digit) + "/maxpool1_c", 6, batch_string)
    combine_channels(ims, tit, nrows, ncols, "maxpool1" + batch_string, digit)

    nrows, ncols = 4, 4
    ims, tit = get_images("feature_maps/" + str(digit) + "/maxpool2_c", 16, batch_string)
    combine_channels(ims, tit, nrows, ncols, "maxpool2" + batch_string, digit)


def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


def plot_tsne(dimg, dlabel, name):

    #cwd = os.getcwd()
    #dataset = LoadMNISTdata(cwd)
    #dataset.loadData()
    #dimg = dataset.test_img[range(0,5000)]
    #dlabel = dataset.test_label[range(0,5000)]

    X = np.vstack([dimg[dlabel==i] for i in range(10)])
    y = np.hstack([dlabel[dlabel==i] for i in range(10)])
    digits_proj = TSNE(random_state=RS).fit_transform(X)
    scatter(digits_proj, y)
    plt.savefig(name, dpi=120)


if __name__ == '__main__':
    #plot_tsne()
    #merge_images("seven")
    pass
