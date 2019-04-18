import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def showimg(image, ax=None, title=None):
    ''' image is a numpy.ndarray '''
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.set_axis_off
    if title is None:
        title = image.shape
    ax.set_xlabel(image.shape)
    ax.set_title(title)
    # plt.pause(0.001)
    return ax

def showsim(simimgs, figsize=(16, 12)):
    num = len(simimgs)
    if num == 1:
        simimg = simimgs[0]
        im = Image.open(simimg['path'])
        showimg(np.asarray(im), title='Sim={} {}'.format(simimg['sim'], simimg['path']))
        return

    _, axs = plt.subplots(1, num, figsize=figsize)
    for i in range(num):
        simimg = simimgs[i]
        im = Image.open(simimg['path'])
        showimg(np.asarray(im), ax=axs[i], title='Sim={} {}'.format(simimg['sim'], simimg['path']))
