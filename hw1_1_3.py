#!/usr/bin/env python3
'''Script to run img_mix on several images for Question 1.3 of HW1'''
import cv2
from image_processing import image_mix


# import images
robot_train_files = ['image15.png', 'image16.png']
bkgd_train_files = ['1.jpg', '15.jpg']
robot_test_files = ['image19.png', 'image20.png']
bkgd_test_files = ['25.jpg', '28.jpg']

# load images and save
def load_images(filenames):
    '''function to load all the images'''
    images = []
    for file in filenames:
        img = cv2.imread(file)
        if img is None:
            print('Error: could not load file')
        else:
            images.append(img)
    return images

# call function to load all images
robot_train_imgs = load_images(robot_train_files)
bkgd_train_imgs = load_images(bkgd_train_files)
robot_test_imgs = load_images(robot_test_files)
bkgd_test_imgs = load_images(bkgd_test_files)

# define thresholds
thresh_low = (0, 100, 0)
thresh_high= (150, 250, 150)

# run image_mix on all combinations
# training images
count = 1
for bkgd in bkgd_train_imgs:
    for robo in robot_train_imgs:
        new_img = image_mix(robo, bkgd, thresh_low, thresh_high)
        cv2.imwrite(f"mixed_training_img{count}.png", new_img)
        count += 1

# testing images
count = 1
for bkgd in bkgd_test_imgs:
    for robo in robot_test_imgs:
        new_img = image_mix(robo, bkgd, thresh_low, thresh_high)
        cv2.imwrite(f"mixed_testing_img{count}.png", new_img)
        count += 1
