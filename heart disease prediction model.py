import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import pathlib 
import utils
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError ,BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("C:\\Users\\Ahmed Salah\\Desktop\\private MO\\programming\\projects\\ML-DL projects\\Diseases prediction model\\dataset\\disease_prediction _edited.csv")
X = df.drop(columns=["result"])
Y = df["result"]
""" male = 1 
    female = 0
    yes = 1
    no = 0
    physical_activiy in the dataset
          low = 0
          medium = 1
          high = 2
"""
random_state = 42
x_train ,x_valid ,x_test ,y_train ,y_valid ,y_test = utils.data_splitting(X ,Y ,0.8 ,0.1 ,0.1 ,random_state=random_state)


scaler = StandardScaler()
scaled_columns = ['age', 'glucose_mg_dl', 'cholesterol_mg_dl', 'systolic_bp', 'diastolic_bp', 'bmi', 'MAP', 'RPP Rate Pressure Product', 'PP Pulse Pressure', 'Atherogenic Index Coefficient', 'Smoking-Hypertension Interaction', 'Cardiac Adiposity Proxy', 'Cardiovascular Stress Index']
unimportant_cols = ["gender" , "alcohol_consumption" , "heart_rate"]
final_x_train = utils.scalingfortrain(x_train , scaled_columns , scaler ,unimportant_cols)
final_x_val = utils.scalingfortest(x_valid , scaled_columns , scaler ,unimportant_cols)
final_x_test = utils.scalingfortest(x_test , scaled_columns , scaler ,unimportant_cols)


model = Sequential([
    Dense(64, input_shape=(final_x_train.shape[1],), activation='relu' ,kernel_regularizer=l2(0.01)),
    Dropout(0.1),
    Dense(16, input_shape=(64,) , activation='relu' ,kernel_regularizer=l2(0.01)),
    Dropout(0.1),
    Dense(32, input_shape=(16,) ,activation='relu' ,kernel_regularizer=l2(0.01)),
    Dropout(0.01),
    Dense(8 ,input_shape=(32,) , activation='relu',kernel_regularizer=l2(0.01)),
    Dropout(0.1),
    Dense(1 ,input_shape=(8,), activation='sigmoid')
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss=BinaryCrossentropy(from_logits=True), 
    metrics=['accuracy', tf.keras.metrics.Recall()] 
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

history = model.fit(
    final_x_train, y_train,
    validation_data=(final_x_val, y_valid),
    epochs=130,       
    batch_size=4, 
    callbacks=[early_stop],     
    verbose=1
)

print("Model evaluation on test data:")
test_loss, test_accuracy, test_recall = model.evaluate(final_x_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Recall: {test_recall:.4f}")


BASE_DIR = pathlib.Path("Diseases prediction model").resolve().parent.parent
model_path = BASE_DIR / "Diseases prediction model" /"saved model" / "heart model.joblib"
joblib.dump(model, model_path)
print(f"the model has saved in: {model_path}")
joblib.dump(scaler, BASE_DIR / "Diseases prediction model" / "saved model" / "Heart_scaler.joblib")
