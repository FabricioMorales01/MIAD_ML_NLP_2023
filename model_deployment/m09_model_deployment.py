#!/usr/bin/python

from transformers import AutoTokenizer, TFAutoModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import os


# Variables globales
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
        'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
        'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']

# Creación del modelo
bert_model = TFAutoModel.from_pretrained("bert-base-uncased")
bert_model.trainable = False

# Creamos la red neuronal
input_ids = tf.keras.layers.Input(shape=(512,), dtype=tf.int32)
token_type_ids = tf.keras.layers.Input(shape=(512,), dtype=tf.int32)
attention_mask = tf.keras.layers.Input(shape=(512,), dtype=tf.int32)

# Nueva capa de entrada para la columna "Year"
year_input = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)

embedding = bert_model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]

# Agregar capas adicionales para clasificación
x = tf.keras.layers.GlobalAveragePooling1D()(embedding)

# Combina la salida de BERT con la entrada "Year"
x = tf.keras.layers.Concatenate()([x, year_input])

x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)  # Aumentar el número de unidades
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
output = tf.keras.layers.Dense(len(cols), activation='sigmoid')(x)


# Crear el modelo
model = tf.keras.Model(
      inputs=[input_ids, token_type_ids, attention_mask, year_input],
      outputs=output
    )

# Compilar el modelo
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))



# Carga del modelo
model.load_weights(os.path.dirname(__file__) + '/bert-with-title-year.h5')

def predict(title, plot, year):
    X = f'{title} : {plot}'

    new_data = [X]
    new_inputs = tokenizer(new_data, return_tensors='tf', truncation=True, padding=True)
    year_train = np.array([year]).reshape(-1, 1) / 100.0 

    padded_input_ids = pad_sequences(new_inputs['input_ids'], maxlen=512, padding='post')
    padded_token_type_ids = pad_sequences(new_inputs['token_type_ids'], maxlen=512, padding='post')
    padded_attention_mask = pad_sequences(new_inputs['attention_mask'], maxlen=512, padding='post')

    new_predictions = model.predict([padded_input_ids, padded_token_type_ids, padded_attention_mask, year_train])

    genres = [cols[i] for i in range(len(cols))  if new_predictions[0, i] > 0.5 ]
    return ', '.join(genres)

        