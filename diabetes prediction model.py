import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from utils import *
import pathlib
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError ,BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
values ={
    "never":0,
    "not current":1,
    "current":3,
    "former":2,
    "ever":1,
    "No Info":1
}

df = pd.read_csv("C:\\Users\\Ahmed Salah\\Desktop\\private MO\\programming\\projects\\ML-DL projects\\Diseases prediction model\\dataset\\diabetes_prediction_dataset.csv")

df['gander'] = replace_values_in_csv(df ,'gender' ,"Female" ,"Male" ,"Other")
df['smoking_history'] = df['smoking_history'].replace(values).astype(int)

X = df.drop(columns=['diabetes'])
Y = df['diabetes']
x_train ,x_val ,x_test ,y_train ,y_val ,y_test = data_splitting(X ,Y ,0.8 ,0.2 ,0.1 ,42)

scaler = StandardScaler()
scaled_cols = ['age' ,'smoking_history' ,'bmi' ,'HbA1c_level' ,'blood_glucose_level']
final_x_train = scalingfortrain(x_train ,scaled_cols ,scaler ,None)
final_x_valid = scalingfortest(x_val ,scaled_cols ,scaler ,None)
final_x_test = scalingfortest(x_test ,scaled_cols ,scaler ,None)

model = Sequential([
    Dense(16 ,input_shape=(final_x_train.shape[1],) ,activation='relu' ,kernel_regularizer=l2(0.001) ,name='l1'),
    Dropout(0.08),
    Dense(8 ,input_shape=(16,) ,activation='relu' ,kernel_regularizer=l2(0.001) ,name='l2'),
    Dropout(0.06),
    Dense(4 ,input_shape=(8,) ,activation='relu' ,kernel_regularizer=l2(0.001) ,name='l3'),
    Dropout(0.04),
    Dense(1 ,input_shape=(4,) ,activation='sigmoid')
])
model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
  metrics=['accuracy' ,tf.keras.metrics.Recall()]
 )
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10 ,
    restore_best_weights=True
)
weights = {0 : 0.1, 1: 0.9}
history = model.fit(
    final_x_train ,y_train ,
    validation_data=(final_x_valid ,y_val),
    epochs= 30,
    class_weight=weights,
    batch_size=128 ,
    callbacks=[early_stop] ,
    verbose=2
)
print("Model evaluation on test data :  ")
test_loss ,test_accuracy ,test_recall = model.evaluate(final_x_test ,y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Recall: {test_recall:.4f}")


BASE_DIR = pathlib.Path("Diseases prediction model").resolve().parent.parent
model_path = BASE_DIR / "Diseases prediction model" / "saved model" / "diabetes model.joblib"
joblib.dump(model , model_path)
joblib.dump(scaler , BASE_DIR / "Diseases prediction model" / "saved model" / "diabetes_scaler.joblib")