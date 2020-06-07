import tensorflow as tf
import cv2
import os
import numpy as np


print("Loading Model...")
model=tf.keras.models.load_model('model_08015_07680.h5')
print("Model Loaded")
print("Preparing test dataset...")
f=open('labels.txt','r')
labels=f.read().split('\n')
y_true=list(map(int,labels[:-1]))
y_true=tf.keras.utils.to_categorical(y_true)
li=os.listdir('images//')
ims=[]
for j in range(len(li)):
    img=cv2.imread('images//'+str(j)+'.jpg')
    img=cv2.resize(img,(200,200))
    ims.append(img)
ims=((np.array(ims)-127.5)/127.5).astype(np.float32)

print("Evaluating...")
res=model.evaluate(ims,y_true)
print("\nval_accuracy: %.4f\nval_loss: %.4f" %(res[1],res[0]))
