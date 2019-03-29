import cv2

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
    orb = cv2.ORB()
    kp, des = orb.detectAndCompute(gray, None)
    img = cv2.drawKeypoints(gray,kp,img)
    output_img_path = '{}_orb.{}'.format(img_file, img_ext)
    cv2.imwrite(output_img_path, img)
    return img, kp, des
