import matplotlib.pyplot as plt

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
    return ax
