from os import listdir, path 
from img_feature import draw_matches, get_matches, img_fingerprint, print_matches, img_similarity as imsim
from helper import showimg

def findAllImg(rootdir):
    images = []
    if not path.isdir(rootdir):
        return images
    dirs = listdir(rootdir)
    for d in dirs:
        dp = path.join(rootdir, d)
        if path.isfile(dp):
            if '.jpg' in dp:
                images.append(dp)
        elif path.isdir(dp):
            images += findAllImg(dp)
    return images

def mat2str(mat):
    row_str_list = list()
    for row in mat:
        row_str = ' '.join(list(map(lambda x:str(x), row)))
        row_str_list.append(row_str)
    return '-'.join(row_str_list)

class ImSeq:
    def __init__(self, imlist):
        self.imlist = imlist
    def seq(self):
        imlist = self.imlist
        seqlist = list()
        for im in imlist:
            seqdict = dict()
            impathlist = im.split('\\')
            seqdict['label'] = impathlist[-2]
            seqdict['path'] = im.replace('\\', '/')
            seqdict['token'] = impathlist[-2] + '-' + impathlist[-1].split('.')[0]
            seqdict['fp'] = mat2str(img_fingerprint(im)[0])
            seqlist.append(seqdict)
        return seqlist

def simrank(ims, idx, threshold = 0):
    imref = ims[idx]
    simlist = list()
    for idx, im in enumerate(ims):
        simdict = dict()
        sim = imsim(imref['path'], im['path'], threshold=threshold)
        if sim > 0:
            simdict['sim'] =  sim/ 500
            simdict['token'] = im['token']
            simdict['path'] = im['path']
            simdict['idx'] = idx
            simlist.append(simdict)
    return sorted(simlist, key = lambda x:x['sim'], reverse = True)

def getAllImg(rootdir='images'):
    images = []
    if not path.isdir(rootdir):
        return images
    dirs = listdir(rootdir)
    for d in dirs:
        dp = path.join(rootdir, d)
        if path.isfile(dp):
            if '.jpg' in dp:
                images.append(dp)
        elif path.isdir(dp):
            images += findAllImg(dp)

    seqlist = list()
    for im in images:
        seqdict = dict()
        impathlist = im.split('\\')
        seqdict['label'] = impathlist[-2]
        seqdict['path'] = im.replace('\\', '/')
        seqdict['token'] = impathlist[-2] + '-' + impathlist[-1].split('.')[0]
        seqdict['fp'] = mat2str(img_fingerprint(im)[0])
        seqlist.append(seqdict)
    return seqlist

def simsearch(img, allimgs=None, imroot='images', threshold = 0):
    if allimgs is None:
        allimgs = getAllImg(imroot)
    simlist = list()
    for idx, im in enumerate(allimgs):
        simdict = dict()
        sim = imsim(img, im['path'], threshold=threshold)
        if sim > 0:
            simdict['sim'] =  sim/ 500
            simdict['path'] = im['path']
            simlist.append(simdict)
    return sorted(simlist, key = lambda x:x['sim'], reverse = True)