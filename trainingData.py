import numpy as np
import cv2

import numpy as np
import cv2
from PIL import Image

from shutil import copyfile
import shutil

import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Process
import queue
import threading
import urllib.request as urllib2


TRAINING_DATASET = 'training_dataset'
TRAINING_IMAGES = TRAINING_DATASET + '/images'
IMG_EXT = '.png'
TRAINING_CSV = TRAINING_DATASET + '/boneage-training-dataset.csv'

FROM = 'M:/Desktop/bone age/boneage-training-dataset/boneage-training-dataset'
TO = TRAINING_DATASET + '/M'

TO2 = TRAINING_DATASET + '/M_labeled_augmented'

import csv
from image import *
import os

trainingLabels = []
with open(TRAINING_CSV, newline='') as csvfile:
    stream = csv.DictReader(csvfile)
    for row in stream:
        trainingLabels.append(row)


def getBoneAgeById(filename):
    for row in trainingLabels:
        if row['id'] == filename:
            return row['boneage']
    return -1

def getGenderById(filename):
    for row in trainingLabels:
        if row['id'] == filename:
            return row['male']
    return -1


def preprocessSingleImage(filename, preprocess=0):
    id, file_extension = filename.split(".")
    if getGenderById(id) == 'False':
        return
    boneage = getBoneAgeById(id)
    if not os.path.exists(TO2 + '/' + str(boneage)):
        os.makedirs(TO2 + '/' + str(boneage))

    # im = Image.open(TO + '/' + filename)
    im = cv2.imread(FROM + '/' + filename, cv2.IMREAD_GRAYSCALE)
    # -- prepare image
    if preprocess:
        im = preprocessImage(im)
    im = scaleImage(im, 500)
    im = Image.fromarray(im)
    im.save(TO2 + '/' + str(boneage) + '/' + id + '.jpg', 'JPEG', quality=100)
    # Image.new("L", im.size, )
    # copyfile(TO + '/' + filename, TRAINING_DATASET + '/' + 'M_labeled/' + str(boneage) + '/' + filename)
    print(id + " saved")


class TrainingData:

    def __init__(self):
        self.image = ImageData(1377)
        print(self.image.gender)


    def copyTrainingImages(self, gender=1, minAge=60, maxAge=180):

        if not os.path.exists(TO):
            os.makedirs(TO)

        gender = True if gender == 1 else False
        for row in trainingLabels:
            if row['male'] == str(gender) and minAge <= int(row['boneage']) <= maxAge:
                print('found', row['id'], row['male'], row['boneage'])
                copyfile(FROM+'/'+ row['id']+IMG_EXT, TO+'/'+ row['id']+IMG_EXT)

    def putToLabelFolders(self):

        # if not os.path.exists(TRAINING_DATASET + '/' + 'M_labeled'):
        #     os.makedirs(TRAINING_DATASET + '/' + 'M_labeled')




        pool = ThreadPool(4)
        results = pool.map(preprocessSingleImage, os.listdir(FROM))

        pool.close()
        pool.join()


        # for filename in os.listdir(FROM):
        #     p = Process(target=preprocessSingleImage, args=(filename,))
        #     p.start()
        #     p.join()

        # for filename in os.listdir(FROM):
        #     t = threading.Thread(target=preprocessSingleImage, args=(filename,))
        #     t.daemon = True
        #     t.start()


        # results = []
        # for item in my_array:
        #     results.append(my_function(item))

        # for filename in os.listdir(FROM):
        #     preprocessSingleImage(filename)

    def copytree(self, src, dst, symlinks=False, ignore=None):
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, symlinks, ignore)
            else:
                shutil.copy2(s, d)


    def removeSmallLabels(self, dir='training_dataset/m_augmented/'):
        dirs_to_remove = []
        for root, dirs, files in os.walk(dir):

            # print(root)
            # print(dirs)
            #print("in "+dirs+' are '+ str(len(files)) + ' files')
            for d in dirs:
                for r, f, fi in os.walk(root+d):
                    # print("x", r, f, len(fi), root, d)
                    if len(fi) < 35:
                        dirs_to_remove.append(d)
                        if not os.path.exists('training_dataset/copied_small/' + d):
                            os.makedirs('training_dataset/copied_small/' + d)
                        # shutil.copy(root+d, 'training_dataset/copied_small/' + d)
                        self.copytree(root+d, 'training_dataset/copied_small/' + d)


        print("dirs to remove")
        # print(dirs_to_remove)
        for d in dirs_to_remove:
        #     print("del", d, dir)
        #     if not os.path.exists('training_dataset/copied_small/'+d):
        #         os.makedirs('training_dataset/copied_small/'+d)

            # for root, dirs, f in os.walk(dir):
            # for files in dir+d:
            #     shutil.copyfile(dir+d+'/'+f, 'training_dataset/copied_small/'+d+'/'+f)

            #shutil.copy(dir+d, 'training_dataset/copied_small')
            shutil.rmtree(dir + '/' + d)
            #print('dir '+ d + ' removed')

    def trainsetAugmentation(self, dirFrom, dirTo):
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            width_shift_range=0.2,
            height_shift_range=0.2,
            rotation_range=20,
            zoom_range=0.2,
            horizontal_flip=True)

        # for root, dirs, files in os.walk(dir):
        #     for d in dirs:

        train_generator = train_datagen.flow_from_directory(
           dirFrom,
            target_size=(500, 500),
            color_mode='grayscale',
            batch_size=32,
            class_mode='binary',
            save_to_dir=dirTo,
            save_format='jpeg')

        i = 0
        for batch in train_generator:

            i+=1
            if i > len(train_generator.filenames):
                break


td = TrainingData()
# td.copyTrainingImages()
#td.putToLabelFolders()
#td.removeSmallLabels()
td.trainsetAugmentation('training_dataset/M_labeled0/', 'training_dataset/M_labeled_augmented/')