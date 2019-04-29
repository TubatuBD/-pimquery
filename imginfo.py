from os import listdir, path, remove
from PIL import Image
import re

def allimgs(root):
    imgs = [path.join(root, cur).replace('\\', '/') for cur in listdir(root) if path.isfile(path.join(root, cur)) and re.search(r'.(jpg|png|jpeg)$', cur)]
    dirs = [path.join(root, cur) for cur in listdir(root) if path.isdir(path.join(root, cur))]
    if len(dirs) == 0:
        return imgs
    for curdir in dirs:
        imgs += allimgs(curdir)
    return imgs

if __name__ == '__main__':
    print(allimgs('images/train'))
