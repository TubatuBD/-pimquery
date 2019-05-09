import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import json

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

def showimg_by_path(imgs):
    _, axs = plt.subplots(1, len(imgs), figsize=(16, 12))
    for i in range(len(imgs)):
        im = np.asarray(Image.open(imgs[i]))
        showimg(im, ax=axs[i], title=imgs[i])

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

def saveJson(data, json_file):
    save_json = json.dumps(data)
    with open(json_file, 'w') as f:
        f.write(save_json)

def loadJson(json_file):
    with open(json_file, 'r') as f:
        res = json.loads(f.readlines()[0])
    return res