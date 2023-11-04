import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import warnings
import re
warnings.filterwarnings('ignore')
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

os.chdir('C:/Users/dscha/Downloads/D214/Diabetes')

df=pd.read_csv('diabetes_prediction_dataset.csv')

print(df.head())

df['diabetes'].value_counts()

df['male']=[
    1 if gender=='Male' else 0 for gender in df['gender']
]
df.drop('gender', axis=1, inplace=True)

df['smoker']=[
    0 if smoke=='never'
    else 1 if smoke=='current'
    else 2 if smoke=='not current' or smoke=='former'
    else 3 for smoke in df['smoking_history']
]
    

df.drop('smoking_history', axis=1, inplace=True)

df.isnull().any()

X=df.drop('diabetes', axis=1)
y=df['diabetes']

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

X_train, X_test, y_train, y_test=train_test_split(X_scaled,y,test_size=0.2, random_state=42, stratify=y)

tf.random.set_seed(42)
model=Sequential()
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss=binary_crossentropy, optimizer=tf.keras.optimizers.Adam(lr=0.03), metrics=[BinaryAccuracy(name='accuracy'), Precision(name='precision'), Recall(name='recall')])

#stopping=EarlyStopping(monitor='accuracy', patience=5)
save_weights=ModelCheckpoint(
    filepath='BestWeights.hdf5',
    monitor='recall',
    mode='max',
    save_weights_only=False,
    save_best_only=True,
    verbose=1
)
history=model.fit(X_train, y_train, epochs=100, callbacks=[save_weights])
model.summary()

model.load_weights('BestWeights100-100-100.hdf5')
predictions=model.predict(X_test)

prediction_classes=[
    1 if prob>0.5 else 0 for prob in np.ravel(predictions)
]

print(prediction_classes[:20])

print(confusion_matrix(y_test, prediction_classes))

print(f'Accuracy: {accuracy_score(y_test, prediction_classes):.2f}')
print(f'Precision:{precision_score(y_test, prediction_classes):.2f}')
print(f'Recall:{recall_score(y_test, prediction_classes):.2f}')