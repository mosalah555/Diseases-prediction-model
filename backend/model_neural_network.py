import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
#data preprocessing and splitting the data into training, validation, and test sets
df = pd.read_csv('C:\\Users\\Ahmed Salah\\Desktop\\private MO\\programming\\projects\\ML-DL projects\\Heart attack and diseases model\\backend\\database\\disease_prediction _edited.csv')
""" male = 1 
    female = 0
    yes = 1
    no = 0
    physical_activiy in the dataset
          low = 0
          medium = 1
          high = 2
"""
X = df.drop(columns=['result']) 
Y = df['result']
scaler = StandardScaler()
x_train ,x_temp ,y_train ,y_temp = train_test_split(
    X ,Y,
    test_size=0.3,
    random_state=42,
)
x_val ,x_test ,y_val ,y_test = train_test_split(
    x_temp ,y_temp,
    test_size=0.5,
    random_state=42,
)
#fitting the scaler on the training data and transforming the training, validation, and test sets
unimportant_cols = ['gender', 'alcohol_consumption', 'heart_rate']
scaled_x_train = scaler.fit_transform(x_train[['age', 'glucose_mg_dl', 'cholesterol_mg_dl', 'systolic_bp', 'diastolic_bp', 'bmi', 'MAP', 'RPP Rate Pressure Product', 'PP Pulse Pressure', 'Atherogenic Index Coefficient', 'Smoking-Hypertension Interaction', 'Cardiac Adiposity Proxy', 'Cardiovascular Stress Index']])
nonscaled_x_train = x_train.drop(columns=['age', 'glucose_mg_dl', 'cholesterol_mg_dl', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'bmi', 'MAP', 'RPP Rate Pressure Product', 'PP Pulse Pressure', 'Atherogenic Index Coefficient', 'Smoking-Hypertension Interaction', 'Cardiac Adiposity Proxy', 'Cardiovascular Stress Index']  + unimportant_cols)
scaled_x_val = scaler.transform(x_val[['age', 'glucose_mg_dl', 'cholesterol_mg_dl', 'systolic_bp', 'diastolic_bp', 'bmi', 'MAP', 'RPP Rate Pressure Product', 'PP Pulse Pressure', 'Atherogenic Index Coefficient', 'Smoking-Hypertension Interaction', 'Cardiac Adiposity Proxy', 'Cardiovascular Stress Index']])
nonscaled_x_val = x_val.drop(columns=['age', 'glucose_mg_dl', 'cholesterol_mg_dl', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'bmi', 'MAP', 'RPP Rate Pressure Product', 'PP Pulse Pressure', 'Atherogenic Index Coefficient', 'Smoking-Hypertension Interaction', 'Cardiac Adiposity Proxy', 'Cardiovascular Stress Index'] + unimportant_cols)
scaled_x_test = scaler.transform(x_test[['age', 'glucose_mg_dl', 'cholesterol_mg_dl', 'systolic_bp', 'diastolic_bp', 'bmi', 'MAP', 'RPP Rate Pressure Product', 'PP Pulse Pressure', 'Atherogenic Index Coefficient', 'Smoking-Hypertension Interaction', 'Cardiac Adiposity Proxy', 'Cardiovascular Stress Index']])
nonscaled_x_test = x_test.drop(columns=['age', 'glucose_mg_dl', 'cholesterol_mg_dl', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'bmi', 'MAP', 'RPP Rate Pressure Product', 'PP Pulse Pressure', 'Atherogenic Index Coefficient', 'Smoking-Hypertension Interaction', 'Cardiac Adiposity Proxy', 'Cardiovascular Stress Index'] + unimportant_cols)
final_x_train = np.concatenate((scaled_x_train, nonscaled_x_train), axis=1)
final_x_val = np.concatenate((scaled_x_val, nonscaled_x_val), axis=1)
final_x_test = np.concatenate((scaled_x_test, nonscaled_x_test), axis=1)
#model building and training using a Sequential model with Dense layers and Dropout for regularization
model = Sequential([
    Dense(16 ,input_shape=(final_x_train.shape[1],) ,activation='relu', kernel_regularizer=l2(0.01), name=("L1")),
    Dropout(0.4),
    Dense(4 ,input_shape=(16,) ,activation='relu',kernel_regularizer=l2(0.01), name=("L2")),
    Dropout(0.3),
    Dense(1 ,input_shape=(4,) ,activation='sigmoid',kernel_regularizer=l2(0.01), name=("output"))
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss=BinaryCrossentropy(from_logits=True), 
    metrics=['accuracy', tf.keras.metrics.Recall()] 
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    final_x_train, y_train,
    validation_data=(final_x_val, y_val),
    epochs=100,       
    batch_size=4, 
    callbacks=[early_stop],     
    verbose=1
)
print("Model evaluation on test data:")
test_loss, test_accuracy, test_recall = model.evaluate(final_x_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Recall: {test_recall:.4f}")
BASE_DIR = Path("Heart attack and diseases model").resolve().parent.parent
model_path = BASE_DIR / "backend" / "NN_model.joblib"
joblib.dump(model, model_path)
print(f"the model has saved in: {model_path}")
joblib.dump(scaler, BASE_DIR / "backend" / "NN_scaler.joblib")
model.save('NN_model.keras')
