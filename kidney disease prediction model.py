import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from utils import *
import pathlib
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError ,BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
'''
normal = 0, abnormal = 1
not_present = 0, present = 1
no = 0 ,yes = 1
good = 0 ,poor = 1
low = 0, moderate = 1, high = 2
no_disease = 0

'''
df = pd.read_csv("C:\\Users\\Ahmed Salah\\Desktop\\private MO\\programming\\projects\\ML-DL projects\\Diseases prediction model\\dataset\\kidney_disease_dataset.csv")

df['Red blood cells in urine'] = replace_values_in_csv(df ,'Red blood cells in urine' ,"normal" ,"abnormal" ,value_2=None)
df['Pus cells in urine'] = replace_values_in_csv(df ,'Pus cells in urine' ,"normal" ,"abnormal" ,value_2=None)
df['Pus cell clumps in urine'] = replace_values_in_csv(df ,'Pus cell clumps in urine' ,"not present" ,"present" ,value_2=None)
df['Bacteria in urine'] = replace_values_in_csv(df ,'Bacteria in urine' ,"not present" ,"present" ,value_2=None)
df['Hypertension (yes/no)'] = replace_values_in_csv(df ,'Hypertension (yes/no)' ,"no" ,"yes" ,value_2=None)
df['Diabetes mellitus (yes/no)'] = replace_values_in_csv(df ,'Diabetes mellitus (yes/no)' ,"no" ,"yes" ,value_2=None)
df['Coronary artery disease (yes/no)'] = replace_values_in_csv(df ,'Coronary artery disease (yes/no)' ,"no" ,"yes" ,value_2=None)
df['Appetite (good/poor)'] = replace_values_in_csv(df ,'Appetite (good/poor)' ,"good" ,"poor" ,None)
df['Pedal edema (yes/no)'] = replace_values_in_csv(df ,'Pedal edema (yes/no)' ,"no" ,"yes" ,value_2=None)
df['Anemia (yes/no)'] = replace_values_in_csv(df ,'Anemia (yes/no)' ,"no" ,"yes" ,value_2=None)
df['Family history of chronic kidney disease'] = replace_values_in_csv(df ,'Family history of chronic kidney disease' ,"no" ,"yes" ,value_2=None)
df['Smoking status'] = replace_values_in_csv(df ,'Smoking status' ,"no" ,"yes" ,value_2=None)
df['Physical activity level'] = replace_values_in_csv(df ,'Physical activity level' ,"low" ,"moderate" ,"high")
df['Urinary sediment microscopy results'] = replace_values_in_csv(df ,'Urinary sediment microscopy results' ,"normal" ,"abnormal" ,value_2=None)

risk_mapping = {
    "No_Disease": 0,
    "Low_Risk": 0,
    "Moderate_Risk": 1,
    "High_Risk": 2,
    "Severe_Disease": 2
    }

df["Target"] = df["Target"].replace(risk_mapping).astype(int)

X = df.drop(columns=['Target'])
Y = df['Target']
x_train ,x_valid ,x_test ,y_train ,y_valid ,y_test = data_splitting(X ,Y ,0.7 ,0.2 ,0.1 ,42)
scaler = StandardScaler()
scaled_cols = ['Age of the patient' ,'Blood pressure (mm/Hg)' ,'Specific gravity of urine' ,'Albumin in urine' ,'Sugar in urine' ,'Random blood glucose level (mg/dl)' ,'Blood urea (mg/dl)' ,'Serum creatinine (mg/dl)' ,'Sodium level (mEq/L)' ,'Potassium level (mEq/L)' ,'Hemoglobin level (gms)' ,'Packed cell volume (%)' ,'White blood cell count (cells/cumm)' ,'Red blood cell count (millions/cumm)' ,'Estimated Glomerular Filtration Rate (eGFR)' ,'Urine protein-to-creatinine ratio' ,'Urine output (ml/day)' ,'Serum albumin level' ,'Cholesterol level' ,'Parathyroid hormone (PTH) level' ,'Serum calcium level' ,'Serum phosphate level' ,'Body Mass Index (BMI)' ,'Physical activity level' ,'Duration of diabetes mellitus (years)' ,'Duration of hypertension (years)' ,'Cystatin C level' ,'C-reactive protein (CRP) level' ,'Interleukin-6 (IL-6) level' ]
final_x_train = scalingfortrain(x_train ,scaled_cols ,scaler ,None)
final_x_val = scalingfortest(x_valid ,scaled_cols ,scaler ,None)
final_x_test = scalingfortest(x_test ,scaled_cols ,scaler ,None)
y_train = to_categorical(y_train, num_classes=3)
y_valid = to_categorical(y_valid, num_classes=3)
y_test  = to_categorical(y_test, num_classes=3)

model = Sequential([
    Dense(45, input_shape=(42,) ,activation='relu' ,kernel_regularizer=l2(0.01) ,name='l1'),
    Dropout(0.06),
    Dense(30 ,input_shape=(45,) ,activation='relu' ,kernel_regularizer=l2(0.01) ,name='l2'),
    Dropout(0.04),
    Dense(15 ,input_shape=(30,) ,activation='relu' ,kernel_regularizer=l2(0.01) ,name='l3'),
    Dropout(0.02),
    Dense(3 ,input_shape=(15,) ,activation='softmax')
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), 
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
    epochs=15,       
    batch_size=64, 
    callbacks=[early_stop],     
    verbose=2
)

print("Model evaluation on test data:")
test_loss, test_accuracy, test_recall = model.evaluate(final_x_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Recall: {test_recall:.4f}")

BASE_DIR = pathlib.Path("Diseases prediction model").resolve().parent.parent
model_path = BASE_DIR / "Diseases prediction model" /"saved model" / "kidney model.joblib"
joblib.dump(model, model_path)
print(f"the model has saved in: {model_path}")
joblib.dump(scaler, BASE_DIR / "Diseases prediction model" / "saved model" / "kidney_scaler.joblib")

