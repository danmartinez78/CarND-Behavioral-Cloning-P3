import csv
import numpy as np
from PIL import Image
import pickle

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

        # add images and angles to data set
        car_images.extend([img_center, img_left, img_right])
        steering_angles.extend([steering_center, steering_left, steering_right])

data = {'features': np.array(car_images), 'labels': np.array(steering_angles)}

with open("./data/data.p", "wb") as f:
    pickle.dump(data, f)

print('DONE')