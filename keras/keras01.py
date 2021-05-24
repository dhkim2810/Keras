import numpy as np
import tensorflow as tf

#1. Prepare data
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. Model configuration
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(30,input_dim=1))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(1))

#3. Compile, Train
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=50, batch_size=1)

#4. Evaluation
loss = model.evaluate(x,y, batch_size=1)
print('loss : ', loss)

results = model.predict([4])
print('results : ', results)

#Assignment
#1. MSE
#2. Default value for batch_size
#3. Hyperparameter tuning