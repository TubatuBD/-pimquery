from hash import pHash

def HmDist(a, b):
    hm_dis = 0
    for i in range(len(a)):
        a_int = ord(a[i])
        b_int = ord(b[i])
        ab_nor_bin = bin(a_int ^ b_int)
        hm_dis += sum([int(ab_nor_bin[bt]) for bt in range(2, len(ab_nor_bin))])
    return hm_dis

def ImDiff(img1_path, img2_path):
    return HmDist(pHash(img1_path), pHash(img2_path))
