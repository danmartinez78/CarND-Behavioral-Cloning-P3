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
        steering = float(row[3])
        steering_flipped = -steering

        # read in images from center camera
        path = "./data/"  # fill in the path to your training IMG directory
        img_center = np.asarray(Image.open(path + row[0]))
        img_flipped =  np.fliplr(img_center)

        # add images and angles to data set
        images = [img_center, img_flipped]
        car_images.extend(images)
        angles = [steering, steering_flipped]
        steering_angles.extend(angles)

data = {'features': np.array(car_images), 'labels': np.array(steering_angles)}

with open("./data/data.p", "wb") as f:
    pickle.dump(data, f)

print('DONE')