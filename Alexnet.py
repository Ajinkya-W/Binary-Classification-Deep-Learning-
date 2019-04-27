#!/usr/bin/env python
# coding: utf-8

# In[1]:


import SimpleITK as sitk
import os
import numpy as np
#import matplotlib.pyplot as plt


# In[2]:


import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, Callback

# In[4]:


def find_images(path_dir):
    numb = 50
    X_train=np.zeros((512*numb,240,240),dtype=np.uint16)
    Y_train=np.zeros((512*numb,2),dtype=np.uint16)
    j=0
    for item in os.listdir(path_dir):
        item=os.path.join(path_dir,item)
        for item2 in os.listdir(item):
            im ={'T1':None,'gt':None}
            item2=os.path.join(item,item2)
            for item3 in os.listdir(item2):
                item3=os.path.join(item2,item3)
                for item4 in os.listdir(item3):
                    item5=os.path.join(item3,item4)
                    if os.path.isfile(item5) and item5.endswith('.mha'):
                        itk_image = sitk.ReadImage(item5)
                        nd_image = sitk.GetArrayFromImage(itk_image)
                        if 'more' in item5 or 'OT' in item5:
                            im['gt']=nd_image
                        elif 'T1' in item5 and 'T1c' not in item5:
                            im['T1']=nd_image
            for i in range(55,55+numb):
                if not sum(sum(im['gt'][i,:,:])):
                    for k in range(15):
                        if j >= 512*numb:
                            break
                        Y_train[j][0]=1
                        X_train[j]=im['T1'][i,:,:]
                        j+=1
                        
                else:
                    if j >= 512*numb:
                        break
                    Y_train[j][1]=1
                    X_train[j]=im['T1'][i,:,:]
                    j+=1
                if j >= 512*numb:
                    break
                    
    return X_train,Y_train

# In[5]:


path_dir="../BRATS2015_Training"
X_train,Y_train=find_images(path_dir)


# In[15]:


shape=X_train.shape
X_train=X_train.reshape(shape[0],shape[1],shape[2],1)
X_train.shape


# In[16]:


np.random.seed(1000)
model=Sequential()

model.add(Conv2D(filters=96, input_shape=(240,240,1), kernel_size=(11,11), strides=(4,4), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))
model.add(BatchNormalization())

model.add(Conv2D(filters=256,kernel_size=(11,11),strides=(1,1),padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))
model.add(BatchNormalization())

model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())

model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())

model.add(Dense(2))
model.add(Activation('sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[18]:


model_checkpoint = ModelCheckpoint('./Alexnet_brat.hdf5', monitor='loss',verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.000001, verbose=1)
callbacks = [reduce_lr, model_checkpoint]
model.load_weights("../run1/Alexnet_brat.hdf5")
model.fit(X_train,Y_train,batch_size=64,epochs=100,verbose=1,validation_split=0.2,shuffle=True, callbacks=callbacks)


# In[12]:


#plt.imshow(X_train[13699])


# In[14]:


# Y_train[13699]

