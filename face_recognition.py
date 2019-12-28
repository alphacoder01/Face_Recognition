'''
This file uses a LBPH face recognizer model which is trained on the face data generated using face_data_generate.py

'''

# Import necessary modules
import os
from PIL import Image
import numpy as np
import pickle
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,'face_data')

current_id = 0
label_ids = {}
y_labels = []
x_train = []

recognizer = cv2.face.LBPHFaceRecognizer_create()

# convert the images to a numpy array and create labels for each person
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('.jpg'):
            path = os.path.join(root, file)
            label = os.path.basename(root)
            # print(label, path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_  = label_ids[label]
            # print(label_ids)

            pil_image = Image.open(path)
            image_array = np.array(pil_image, 'uint8')
            x_train.append(image_array)
            y_labels.append(id_)

# save labels as a pickle file
with open('labels.pickle','wb') as f:
    pickle.dump(label_ids,f) 

# train the model
recognizer.train(x_train,np.array(y_labels))
# save the weights
recognizer.save('trainer.yaml')