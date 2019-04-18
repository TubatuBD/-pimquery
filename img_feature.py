import cv2
import matplotlib.pyplot as plt

def sift(img_path):
    img_file, img_ext = tuple(img_path.split('.'))
    img = cv2.imread(img_path)
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(200)
    kp, des = sift.detectAndCompute(gray, None)
    img = cv2.drawKeypoints(gray,kp,img)
    output_img_path = '{}_shift.{}'.format(img_file, img_ext)
    cv2.imwrite(output_img_path, img)
    return img, kp, des

def surf(img_path):
    img_file, img_ext = tuple(img_path.split('.'))
    img = cv2.imread(img_path)
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create(200)
    kp, des = surf.detectAndCompute(gray, None)
    img = cv2.drawKeypoints(gray,kp,img)
    output_img_path = '{}_surf.{}'.format(img_file, img_ext)
    cv2.imwrite(output_img_path, img)
    return img, kp, des

def orb(img_path):
    img_file, img_ext = tuple(img_path.split('.'))
    img = cv2.imread(img_path)
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(gray, None)
    img = cv2.drawKeypoints(gray,kp,img)
    output_img_path = '{}_orb.{}'.format(img_file, img_ext)
    cv2.imwrite(output_img_path, img)
    return img, kp, des

def img_fingerprint(img_path):
    orb = cv2.ORB_create()
    img = cv2.imread(img_path)
    kp, des = orb.detectAndCompute(img, None)
    return (des, kp, img)

def get_matches(img1_path, img2_path):
    des1, kp1, img1 = img_fingerprint(img1_path)
    des2, kp2, img2 = img_fingerprint(img2_path)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    return matches

def draw_matches(img1_path, img2_path):
    des1, kp1, img1 = img_fingerprint(img1_path)
    des2, kp2, img2 = img_fingerprint(img2_path)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    img_match = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
    plt.imshow(img_match)

def print_matches(matches, num = 5):
    for match in matches[:num]:
        print('distance={} trainIdx={} queryIdx={} imgIdx={}'.format(match.distance, match.trainIdx, match.queryIdx, match.imgIdx))

def img_similarity(img1_path, img2_path, threshold = 0, limit = None):
    matches = get_matches(img1_path, img2_path)
    max_num = len(matches)
    if limit is not None:
        if max_num > limit:
            max_num = limit

    sim = 0
    for match in matches[:max_num]:
        if match.distance <= threshold:
            sim += 1

    return sim
