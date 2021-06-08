from numpy import save
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

#1. Data
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66)

#2. Model
model = Sequential()
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['acc'])

#3. Fit and Train
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
modelpath = './keras/checkpoint/k21_cancer_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(modelpath, monitor='val_loss', save_best_only=True, mode='auto')
hist = model.fit(x_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping, cp])

results = model.evaluate(x_test, y_test)

print('loss : ', results[0])
print('accuracy : ', results[1])

y_pred = model.predict(x_test[-5:-1])
print(y_pred)
print(y_test[-5:-1])

print(hist)
print(hist.history.keys())
print(hist.history['val_loss'])
# print(hist.history['accuracy'])

# 시각화, 그래프
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])

plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss','val_loss'])
plt.show()