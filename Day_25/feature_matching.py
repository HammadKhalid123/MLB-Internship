import cv2 as cv
import numpy as np

def orb_knn_matching(image_path1, image_path2):
    img = cv.imread(image_path1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img2 = cv.imread(image_path2)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    orb = cv.ORB_create(nfeatures=2000)
    keypoints1, descriptors1 = orb.detectAndCompute(gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    bf_knn = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    knn_matches = bf_knn.knnMatch(descriptors1, descriptors2, k=2)

    good_matches_knn = []
    for match_pair in knn_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches_knn.append(m)

    good_matches_knn = sorted(good_matches_knn, key=lambda x: x.distance)

    img_matches_knn = cv.drawMatches(
        img, keypoints1,
        img2, keypoints2,
        good_matches_knn[:30], None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return img_matches_knn, keypoints1, keypoints2, descriptors1, descriptors2, good_matches_knn


def orb_bruteforce_matching(image_path1, image_path2):
    img = cv.imread(image_path1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img2 = cv.imread(image_path2)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    orb = cv.ORB_create(nfeatures=2000)
    keypoints1, descriptors1 = orb.detectAndCompute(gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    bf_cross = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    bf_matches = bf_cross.match(descriptors1, descriptors2)
    bf_matches = sorted(bf_matches, key=lambda x: x.distance)

    img_matches_bf = cv.drawMatches(
        img, keypoints1,
        img2, keypoints2,
        bf_matches[:30], None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return img_matches_bf, keypoints1, keypoints2, descriptors1, descriptors2, bf_matches