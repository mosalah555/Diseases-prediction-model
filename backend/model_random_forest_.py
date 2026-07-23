import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from tensorflow import keras
from sklearn.model_selection import train_test_split , RandomizedSearchCV
from sklearn.metrics import accuracy_score , recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
#data preprocessing and splitting the data into training, validation, and test sets
df = pd.read_csv('C:\\Users\\Ahmed Salah\\Desktop\\private MO\\programming\\projects\\ML-DL projects\\Heart attack and diseases model\\backend\\database\\disease_prediction _edited.csv')
""" male = 1 
    female = 0
    yes = 1
    no = 0
"""
X = df.drop(columns=['result']) 
Y = df['result']
mapping = {"low":0 ,"medium":1 ,"high":2}
X["physical_activity"] = X["physical_activity"].map(mapping)
X = pd.get_dummies(X, columns=["physical_activity"])
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
scaler = StandardScaler()
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
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': [None, 'balanced']
}

rf = RandomForestClassifier(random_state=42)
search = RandomizedSearchCV(rf, param_dist, n_iter=50, cv=5,
                             scoring='recall', n_jobs=-1, random_state=42,verbose=2)
search.fit(final_x_train, y_train)
print(search.best_params_)
best_model = search.best_estimator_

model = RandomForestClassifier(**search.best_params_, random_state=42).fit(final_x_train ,y_train)
y_pred_val = model.predict(final_x_val)
y_pred_test = model.predict(final_x_test)
print(f"Model accuracy on validation data: {accuracy_score(y_pred_val, y_val)}")
print(f"Model recall on validation data: {recall_score(y_pred_val, y_val)}")
print(f"Model accuracy on test data:{accuracy_score(y_pred_test, y_test)}")
print(f"Model recall on test data:{recall_score(y_pred_test, y_test)}")
BASE_DIR = Path("Heart attack and diseases model").resolve().parent.parent
model_path = BASE_DIR / "backend" / "random_forest_model.joblib"
joblib.dump(model, model_path)
print(f"the model has saved in: {model_path}")
joblib.dump(scaler, BASE_DIR / "backend" / "random_forest_scaler.joblib")
