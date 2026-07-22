import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError ,BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler





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