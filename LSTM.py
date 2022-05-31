# -*- coding: utf-8 -*-
"""
Created on Sun May 30 22:17:40 2021

@author: ken95
"""

import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow.keras.layers 
import matplotlib.pyplot as plt


def read_file(file1):
    t3=[]
    #print(file1)
    if (file1.endswith(".csv")):
         try:
            # print(file1)
             f1=open(file1,"r")
             flag=False
             count=0
             for i in f1:
                
                 if flag:
                     flag=False
                 else:
                     t=i.split(",")
                     t1=[]
                     for k in t:
                         k=k.strip()
                         if len(k)>0:
                             t1.insert(len(t1),(float)(k))
                        
                     #t1=t1[1:]
                     #print(t1)
                     #print("2"+str(t2.shape))         
                     t3.insert(0,t1)
                     count=count+1

             if count>150:
                 t3=[]
                 print(file1)
         except IOError as e:
             print("Error while handling files", e)
    #print(len(t3))
    return t3

def reading(yourPath,train_set,label_tr, label):

    allFileList = os.listdir(yourPath)
    for file1 in allFileList:
        t3=[]
        if (file1.endswith(".csv")):
            t3=read_file(yourPath+file1)
            #print(len(t3))
            if len(t3)>0:
                train_set.insert(0,t3)
                label_tr.insert(0,label)
            #print(t3)

    return train_set,label_tr
    
def split(data,label,ratio):
    tr=[]
    te=[]
    l_tr=[]
    l_te=[]
    #randomly swapping data
    stop=len(data)-1
    times=0
    while times<300000:
        i=random.randint(0, stop )
        j=random.randint(0, stop )
        while i==j:
            j=random.randint(0, stop )
        temp=data[i]
        data[i]=data[j]  
        data[j]=temp
        a=label[i]
        label[i]=label[j]
        label[j]=a
        times=times+1
    size=int(len(data)*ratio)
    tr=data[0:size]
    l_tr=label[0:size]
    te=data[size:len(data)]
    l_te=label[size:len(data)]
    return tr,te,l_tr,l_te
    #print(tr,te,l_tr,l_te)
data_set= []
label =[]    

data_set,label=reading('不成功/',data_set,label,0)
print(len(data_set))
data_set,label=reading('成功資料/',data_set,label,1)
print(len(data_set))
#print(len(data_set[0][0]))
tr,te,l_tr,l_te=split(data_set,label,0.95)
#for i in data_set:
    #print(len(i),end=" ")
    #print(len(i[0]))
train_data=np.array(tr)
print(train_data.shape)

print("===>",(train_data.shape))

l_tr=np.array(l_tr)
test_data=np.array(te)
l_te=np.array(l_te)
print("===>",np.shape(train_data))

train_data=train_data.reshape(-1,1,train_data.shape[1]*train_data.shape[2])
print("===>",np.shape(train_data))
print(l_tr.shape)
print("===>",np.shape(test_data))

test_data=test_data.reshape(-1,1,test_data.shape[1]*test_data.shape[2])
print("===>",np.shape(test_data))
print(l_te.shape)


model = Sequential()
model.add(LSTM(500, input_shape=(train_data.shape[1],train_data.shape[2]), return_sequences=True))
model.add(LSTM(500))
model.add(tf.keras.layers.Flatten())
model.add(Dense(1024))
#model.add(tf.keras.layers.Dropout(0.2))
model.add(Dense(1024))
model.add(Dense(2))#2
myadam=tf.keras.optimizers.Adam(
    learning_rate=0.0001, 
    name='adam')
model.compile(optimizer=myadam,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_data, l_tr, epochs=50,  validation_data=(test_data, l_te), verbose=2, shuffle=False)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right') 

test_loss, test_acc = model.evaluate(test_data,  l_te, verbose=2)
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
print(probability_model(test_data))
print(l_te)
