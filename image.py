import numpy as np
import cv2

TRAINING_DATASET = 'training_dataset'
TRAINING_IMAGES = TRAINING_DATASET + '/images'
IMG_EXT = '.png'
TRAINING_CSV = TRAINING_DATASET + '/boneage-training-dataset.csv'

import csv

trainingLabels = []
with open(TRAINING_CSV, newline='') as csvfile:
    stream = csv.DictReader(csvfile)
    for row in stream:
        trainingLabels.append(row)


def scaleImage(img, max_size):
    scaled_image = img
    #print(scaled_image.shape, scaled_image.dtype, scaled_image.size)
    (height, width) = (scaled_image.shape[0], scaled_image.shape[1])
    ratio = height / width
    if width > height:
        width = max_size
        height = int(width * ratio)
    else:
        height = max_size
        width = int(height / ratio)
    scaled_image = cv2.resize(scaled_image, (width, height))
    return scaled_image

def findHand(img):
    histogram = [0] * 256
    #print(histogram)
    (h, w) = img.shape
    for x in range(h):
        for y in range(w):
            histogram[img[x][y]] += 1
    bg_threshold = 0.6 * img.size
    sum = 0
    bg = 0
    for i in range(len(histogram)):
        sum += histogram[i]
        if sum > bg_threshold:
            bg = i
            break
    #print("tresh ", bg)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret, thresh = cv2.threshold(blur, bg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    thresh = cv2.bitwise_not(thresh)

    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print("ccc", len(contours))
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), -1)
    max_contour = contours[0]
    max_area = cv2.contourArea(contours[0])
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # print("area " + str(area))
        if area > max_area:
            max_area = area
            max_contour = cnt
    #print("max area " + str(max_area))
    # cv2.drawContours(img, [max_contour], -1, (255, 0, 0), 3)
    mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(mask, [max_contour], -1, (255, 255, 255), -1)
    output = cv2.bitwise_and(img, mask)
    return output


def preprocessImage(img):
    
    output = findHand(img)
    img1 = cv2.equalizeHist(output)
    return img1


class ImageData:
    def __init__(self, img_id):
        self.id = img_id
        self.path = TRAINING_IMAGES + '/' + str(img_id) + IMG_EXT
        self.img = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)

        for row in trainingLabels:
            if row['id'] == str(img_id):
                print('found', row['id'], row['male'], row['boneage'])
                self.boneage = int(row['boneage'])
                self.gender = 1 if row['male'] is True else 0
                break

        print(self.id, self.boneage, self.gender)

    def getGender(self):
        return 'F' if self.gender == 0 else 'M'

    def getBoneAge(self):
        return str(self.boneage) + ' months ('+str(self.boneage/12)+' years)'

    def getImgPreview(self, max_size=400):

        img = self.img
        preview = scaleImage(img, max_size)
        print(preview.shape, preview.dtype, preview.size)
        return preview

    def getImgPreviewA(self, max_size=400):

        img = self.img

        output = findHand(img)
        img = cv2.equalizeHist(output)

        preview = scaleImage(img, max_size)
        print(preview.shape, preview.dtype, preview.size)
        return preview



    def getImgPreviewB(self, max_size=400):
        img = self.img

        output = findHand(img)
        img1 = cv2.equalizeHist(output)

        img = cv2.GaussianBlur(img1, (3,3), 0)
        # ret, thresh = cv2.threshold(img, 130, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        # thresh = cv2.bitwise_not(thresh)

        img0 = cv2.Canny(img, 150, 200, 3)

        kernel1 = np.ones((5, 5), np.uint8)
        kernel2 = np.ones((10, 30), np.uint8)
        #thresh = cv2.erode(thresh, kernel1, iterations=1 )
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel2)

        hierarchy = []
        im2, contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, hierarchy)
        #contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        # cts = []
        # for i in range(len(contours)):
        #     print("l", len(contours))
        #     print(hierarchy[i])
        #     if (hierarchy[i][3] >= 0):
        #         print("has")
        #         cts.append(contours[i])
        #     print("nope")

        ####
        img = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(img, contours, -1, (255, 0, 0), -1)


        ###
        preview = scaleImage(img0, max_size)
        print(preview.shape, preview.dtype, preview.size)
        return preview

    def getImgPreviewC(self, max_size=400):
        gray = self.img
        ret, thresh = cv2.threshold(gray, 120, 255, 0)
        # contours = cv2.findContours(thresh, 1, 2)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print("ccc", len(contours))

        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        # cv2.drawContours(img, contours, -1, (0, 255, 0), -1)

        max_contour = contours[0]
        max_area = cv2.contourArea(contours[0])
        for cnt in contours:
            cv2.drawContours(img, [cnt], -1, (0, 255, 0), -1)

            area = cv2.contourArea(cnt)
            # print("area "+ str(area))
            if area > max_area:
                max_area = area
                max_contour = cnt
                # approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
                # print(len(approx))
                # ctr = np.array(cnt).reshape((-1, 1, 2)).astype(np.int32)
                # cv2.drawContours(img, approx, 0, (0, 255, 0), -1)

        # print("max area " + str(max_area))
        cv2.drawContours(img, [max_contour], -1, (255, 0, 0), 3)

        preview = scaleImage(img, max_size)
        print(preview.shape, preview.dtype, preview.size)
        return preview

    
