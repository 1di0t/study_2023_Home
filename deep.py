import tensorflow as tf
import pandas as pd
import numpy as np


data = pd.read_csv('gpascore.csv')
data = data.dropna()

y_Data = data['admit'].values
x_Data = []

for i, rows in data.iterrows():
    x_Data.append([rows['gre'], rows['gpa'], rows['rank']])

model = tf.keras.models.Sequential([
     tf.keras.layers.Dense(128 , activation='sigmoid'),
     tf.keras.layers.Dense(256, activation='sigmoid'),
     tf.keras.layers.Dense(1,activation='sigmoid')
    ])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']);

model.fit(np.array(x_Data),np.array(y_Data),epochs=10000);

prediction_Data = model.predict([[990,5.60,2],[400,2.20,1]])

for i in range(0,2):
    print("합격확률 : %0.1f" % (100*np.array(prediction_Data[i])) )
