#Import augmentation libraries
import albumentations as A
import cv2
import numpy as np
from pathlib import Path
#Import libraries for data visualization
import matplotlib.pyplot as plt


import json
#Define a function to read bounding boxes
def get_bbox(data):
    #Create a list to store the bounding boxes
    bboxes = []
    #Loop through the data
    for box in data:
        #Get the bounding box
        bbox = box['bbox']
        bbox.append('damage')
        #Append the bounding box to the list
        bboxes.append(bbox)
    return bboxes


#Define a function read json file
def read_json(json_file:str)->dict:
    """Read a json file.
    Args:
        json_file ([type]): path to the json file
    Returns:
        [dict]: returns a dictionary
    """
    with open(json_file, 'r') as f:
        data = json.loads(f.read())
    return data

#Create a function to load an image
def load_image(img_list:list)->list:
    """Read an image from a file path (JPEG or PNG).
    Args:
        path ([type]): path to the image file
    Returns:
        [list]: returns a list of images
    """
    #Read all the images in the list
    images = []
    for img_path in img_list:
        img = cv2.imread(str(img_path))
        #Check if the image is read correctly
        if img is None:
            print('Failed to read image: {}'.format(path))
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #Add the image to the list
        images.append(img)
    return images


data =  read_json('103_0ac05322-414f-4cfb-9daa-528c4bfad3c3.json')
#Print data annotation
# print(data['annotations'])

#Get the bounding boxes
bboxes = get_bbox(data['annotations'])
#Print the bounding boxes

#Declare an augmentation pipeline
transform = A.Compose([
    #Add crop augmentation
    A.RandomCrop(height=1024, width=1024, p=1.0),
    #Add horizontal flip augmentation
    A.HorizontalFlip(p=0.5),
    #add rotation augmentation
    A.Rotate(limit=5, p=0.5),
],bbox_params=A.BboxParams(format='coco',min_visibility=0.2))


#Define a function to apply the augmentation pipeline
def get_augmented(images:list,bboxes:list,transform:A.Compose)->tuple:
    """Apply the augmentation pipeline to the images and bounding boxes.
    Args:
        images ([type]): list of images
        bboxes ([type]): list of bounding boxes
        transform ([type]): augmentation pipeline
    Returns:
        [tuple]: returns a tuple of images and bounding boxes
    """
    #Create a list to store the augmented images
    augmented_images = []
    #Create a list to store the augmented bounding boxes
    augmented_bboxes = []
    #Loop through the images
    for img, bbox in zip(images, bboxes):
        #Apply the augmentation pipeline
        for idx in range(30):
            augmented = transform(image=img, bboxes=bbox)
            #Append the augmented image to the list
            augmented_images.append(augmented['image'])
            #Append the augmented bounding box to the list
            augmented_bboxes.append(augmented['bboxes'])
    return augmented_images, augmented_bboxes

img = load_image(['103_0ac05322-414f-4cfb-9daa-528c4bfad3c3.jpg'])[0]
#Draw the bounding boxes
# for bbox in bboxes:
#     x, y, w, h,_ = bbox
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)



#Apply the augmentation pipeline
augmented_images, augmented_bboxes = get_augmented([img], [bboxes], transform)

# augmented_image = augmented_images[0]
# augmented_bboxe = augmented_bboxes[0]
#Loop through the augmented images
for idx,(augmented_image, augmented_bbox) in enumerate(zip(augmented_images, augmented_bboxes)):
    #Draw the bounding boxes with id
    
    for bbox in augmented_bbox:
        x, y, w, h,_ = bbox
        #Convert the bounding box to integer
        x, y, w, h = int(x), int(y), int(w), int(h)
        #Define the color of the bounding box
        color = (255, 0, 0)
        cv2.rectangle(augmented_image, (x, y), (x+w, y+h),color, 2)
        cv2.putText(augmented_image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #save the image using idx
    cv2.imwrite('augmented_image_{}.jpg'.format(idx), augmented_image)



# for bbox in augmented_bboxe:
#     #Unpack the bounding box and convert to int
#     x,y,w,h,_=bbox
#     #Convert to int
#     x,y,w,h = int(x),int(y),int(w),int(h)
#     #Define the color of the bounding box 
#     #Red
#     color = (0,0,255)

#     cv2.rectangle(augmented_image, (x, y), (x+w, y+h), color, 2)
# cv2.imwrite('augmented.jpg',augmented_image)