# importing the libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.optimizers import Adam
from sklearn import metrics
from sklearn.model_selection import train_test_split as tts
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import time

print("Libraries are imported....")

# loading the data
data = pd.read_csv("processed.csv")
print("Dataset loaded....")


# dividing the dataset into x and y
print("Dividing into dependent and independent feature....")
y = data['label'].values
x = data.drop('label',axis=1)
y = to_categorical(y)
print(x.shape,y.shape)

print("Dividing the dataset into train and test....")
xtrain, xtest, ytrain, ytest = tts(x,y,test_size=0.2,random_state=42,stratify=y)
print(xtrain.shape,ytrain.shape)
print(xtest.shape,ytest.shape)
print("Dividing is done....")

print("Making the model....")

model = Sequential(name="DigitClass")
model.add(Dense(128,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(10,activation="softmax"))

checkpointer = ModelCheckpoint('DigitAudio.h5', save_best_only=True,monitor='val_loss',mode='auto')

print("Model architecture is done....")
model.compile(loss = "categorical_crossentropy", optimizer="adam",metrics=["accuracy"])
print("Model compiled....")
print("Training is starting....")
start = time.time()
hist = model.fit(xtrain,ytrain,batch_size=16,epochs=200,validation_data=(xtest, ytest),callbacks=[checkpointer])
print("Model training is over....")
print("Total Time taken: ",time.time()-start)
print("Model saved....")

# plotting the figures
print("Plotting the figures....")
plt.figure(figsize=(15,10))
plt.plot(hist.history['accuracy'],c='b',label='train')
plt.plot(hist.history['val_accuracy'],c='r',label='validation')
plt.title("Model Accuracy vs Epochs")
plt.xlabel("EPOCHS")
plt.ylabel("ACCURACY")
plt.legend(loc='lower right')
plt.savefig('./img/accuracy.png')


plt.figure(figsize=(15,10))
plt.plot(hist.history['loss'],c='orange',label='train')
plt.plot(hist.history['val_loss'],c='g',label='validation')
plt.title("Model Loss vs Epochs")
plt.xlabel("EPOCHS")
plt.ylabel("LOSS")
plt.legend(loc='upper right')
plt.savefig('./img/loss.png')
print("Figures saved in the disk....")

model=load_model("DigitAudio.h5")
# testing the model
print("Testing the model....")
print("The result obtained is...\n")
model.evaluate(xtest,ytest)


