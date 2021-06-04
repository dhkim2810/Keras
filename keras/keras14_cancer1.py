import numpy as np
from sklearn.datasets import load_breast_cancer

datasets = load_breast_cancer()

from sklearn.model_selection import train_test_split

x = datasets.data
y = datasets.target
print(x.shape, y.shape)

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Dense(256, input_shape=(30,), activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x,y,batch_size=1,epochs=128,verbose=2,validation_split=0.2)

results = model.evaluate(x,y)
print('loss : ', results[0])
print('metrics : ', results[1])

y_pred = model.predict(x[-5:-1])
print(y_pred)
print(y[-5:-1])