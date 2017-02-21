import csv
import numpy as np
from PIL import Image
import pickle
import cv2

# read in  csv data
csv_file = './data/driving_log.csv'
car_images = []
steering_angles = []
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        correction = 0.5  # this is a parameter to tune
        steering_center = float(row[3])
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # read in images from center, left and right cameras
        path = "./data/"  # fill in the path to your training IMG directory
        img_center = np.asarray(Image.open(path + row[0]))
        img_left = np.asarray(Image.open(path + row[1].lstrip()))
        img_right = np.asarray(Image.open(path + row[2].lstrip()))

        # augment with random brightness and skew
        rows, cols, depth = img_center.shape
        M = np.float32([[1, 0, np.random.randint(-50,50)], [0, 1, np.random.randint(-50,50)]])
        image1 = cv2.warpAffine(img_center, M, (cols, rows))
        image2 = img_center * np.random.random_sample()

        # add images and angles to data set
        car_images.extend([img_center, img_left, img_right, np.fliplr(img_center), image2])
        steering_angles.extend([steering_center, steering_left, steering_right, steering_center])



data = {'features': np.array(car_images), 'labels': np.array(steering_angles)}

with open("./data/data.p", "wb") as f:
    pickle.dump(data, f, protocol=4)

print('DONE')