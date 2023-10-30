import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import losses
from sklearn.model_selection import train_test_split
from tensorflow.keras import metrics
import tensorflow_decision_forests as tfdf
import math
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")

df=pd.read_csv('//home/derrick/diamonds.csv')

pd.DataFrame.duplicated(df) #check for duplicate data

pd.DataFrame.isnull(df).sum() #check for nulls

df=df[['carat','cut','color','clarity','price']]

ohe_cut=pd.get_dummies(df['cut'], dtype=int)

ohe_color=pd.get_dummies(df['color'], dtype=int)

ohe_clarity=pd.get_dummies(df['clarity'], dtype=int)

ohe_cut=ohe_cut.drop('Fair', axis=1)

ohe_color=ohe_color.drop('J', axis=1)

ohe_clarity=ohe_clarity.drop('I1', axis=1)

df=df.drop(['cut', 'color', 'clarity'], axis=1)

df=pd.concat([df, ohe_cut, ohe_color, ohe_clarity], axis=1)

Features=df.drop('price', axis=1)

Target=df['price']

scaler=MinMaxScaler()

Features_normalized = scaler.fit_transform(Features)

X_train, X_save, y_train, y_save = train_test_split(Features_normalized, Target, test_size=.4, random_state=42)

X_test, X_val, y_test, y_val=train_test_split(X_save, y_save, test_size=.5, random_state=42)

train_data=tf.data.Dataset.from_tensor_slices((X_train, y_train))

test_data=tf.data.Dataset.from_tensor_slices((X_test, y_test))

val_data=tf.data.Dataset.from_tensor_slices((X_val, y_val))

train_data=train_data.shuffle(len(df)).batch(1)

test_data=test_data.shuffle(len(df)).batch(1)

val_data=val_data.shuffle(len(df)).batch(1)

model=Sequential()

model.add(Dense(100, 'relu'))

model.add(Dense(10,'relu'))

model.add(Dense(1,None))

training=EarlyStopping(monitor='mae', patience=10)

saving=ModelCheckpoint(
    filepath='//home/derrick/Best_model.hdf5',
    monitor='mae',
    mode='min',
    save_weights_only=False,
    save_best_only=True,
    verbose=1
)

model.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), metrics='mae')

history=model.fit(train_data, epochs=50, validation_data=test_data, callbacks=[training, saving])

model.summary()

model.load_weights('Best_model.hdf5')

predictions=model.predict(X_val)

from sklearn.metrics import mean_absolute_error

mae_predicts = mean_absolute_error(y_val, predictions)

print("Mean Absolute Error:", mae_predicts)

results=pd.DataFrame({"Actual":y_val, "Predicted": predictions.flatten()})

plt.figure(figsize=(10, 6))

plt.scatter(results.index, results["Actual"], label="Actual")

plt.scatter(results.index, results["Predicted"], label="Predicted")

plt.xlabel("Index")

plt.ylabel("Prices")

plt.title("Actual vs. Predicted Prices on Validation Set")

plt.legend()

plt.show()