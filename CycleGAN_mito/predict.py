import keras
import os
import cv2
import numpy as np
from keras.models import load_model

model = load_model('netGB.h5')

train_folder = 'CycleGAN/trainA'
test_folder = 'CycleGAN/testA'

os.mkdir('CycleGAN/train_predicted')
os.mkdir('CycleGAN/test_predicted')

list_all_train_dir = os.listdir(train_folder)
list_all_test_dir = os.listdir(test_folder)
img_list_train = [i for i in list_all_train_dir if i.endswith('.tif')]
img_list_test = [j for j in list_all_test_dir if j.endswith('.tif')]

for idx in range(len(img_list_train)):
    train_name = img_list_train[idx]
    train_path = train_folder+'/'+train_name
    img_train = cv2.imread(train_path,-1)
    img_train = img_train/750.0-1.0
    # print(img_train)
    img_train = img_train.reshape(1,256,256,1)
    img_train_predicted = model.predict(img_train)
    img_train_predicted = img_train_predicted.reshape(256,256)
    img_train_predicted = ((img_train_predicted+1.0)*750.0).astype('uint16')
    cv2.imwrite('CycleGAN/train_predicted/'+train_name,img_train_predicted)
    
for idx in range(len(img_list_test)):
    test_name = img_list_test[idx]
    test_path = test_folder+'/'+test_name
    img_test = cv2.imread(test_path,-1)    
    img_test = img_test/750.0-1.0
    img_test = img_test.reshape(1,256,256,1)    
    img_test_predicted = model.predict(img_test)
    img_test_predicted = img_test_predicted.reshape(256,256)
    img_test_predicted = ((img_test_predicted+1.0)*750.0).astype('uint16')
    cv2.imwrite('CycleGAN/test_predicted/'+test_name,img_test_predicted)