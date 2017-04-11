import csv
import numpy as np
from PIL import Image
import pickle
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# read in  csv data
csv_file = './data/driving_log.csv'
car_images = []
steering_angles = []
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        correction = 0.35  # this is a parameter to tune
        steering_center = float(row[3])
        if abs(steering_center) < 0.001:
            continue
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
n, bins, patches = plt.hist(data['labels'], 100, facecolor = 'green')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.show()
print(len(data['labels']))

with open("./data/data.p", "wb") as f:
    pickle.dump(data, f, protocol=4)

print('DONE')