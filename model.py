import numpy as np 
import pandas as pd 
import cv2
import os 

import keras
from keras.models import Sequential
from keras.utils import to_categorical
from keras import regularizers
from keras.layers import Dense,Conv2D,Dropout,GlobalAveragePooling2D,BatchNormalization

from matplotlib.pyplot import imshow

import tarfile

from sklearn.utils import shuffle
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

import scipy.io




for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


DATA_PATH = '../input/flower-dataset-102/'
PATH = "../working/flower/"

opened_tar = tarfile.open(DATA_PATH+"102flowers.tgz")
opened_tar.extractall(PATH)

def get_all_filenames(tar_fn):
    with tarfile.open(tar_fn) as f:
        return [m.name for m in f.getmembers() if m.isfile()]

df = pd.DataFrame()
df['Id'] = sorted(get_all_filenames(DATA_PATH+"102flowers.tgz"))
df['Category'] = scipy.io.loadmat(DATA_PATH+'imagelabels.mat')['labels'][0] - 1
df['Category'] = df['Category'].astype(str)

df=shuffle(df,random_state=1)

im=[]
y=[]
for k,(i,j) in enumerate(zip(df.Id,df.Category)):
    img=cv2.imread(PATH+i)
    img=cv2.resize(img,(200,200))
    im.append(img)
    y.append(j)
    if k%100==0:
        print(k)
        
im=((np.array(im)-127.5)/127.5).astype(np.float32)
y=np.array(y).reshape((8189,1))
y=to_categorical(y)

x_train,x_test,y_train,y_test=train_test_split(im,y,test_size=0.1,random_state=42)
del im
del y

model=Sequential()
model.add(Conv2D(64,4,input_shape=(200,200,3),strides=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(96,3,activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(96,4,strides=(2,2)))
model.add(Conv2D(96,2,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(128,4,strides=(2,2)))
model.add(Conv2D(128,2,activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(256,3,strides=(2,2),activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(102,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(102,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


datagen = keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
datagen.fit(x_train)

model.fit(x_train,y_train,epochs=100,validation_data=(x_test,y_test))

model.save('model_08015_07680.h5')

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)










