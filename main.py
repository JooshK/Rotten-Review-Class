import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import functools
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    plt.show()

df_train = pd.read_csv("rt_reviews.csv", names=['Freshness', 'Review'], encoding="ISO-8859-1")
df_eval = pd.read_csv("eval.csv", names=["Freshness", "Review"], encoding="ISO-8859-1")

reviews_train = df_train["Review"].values
y_train = np.asarray((df_train["Freshness"].values)).astype(float)
reviews_test = df_eval["Review"].values
y_test = np.asarray((df_eval["Freshness"].values)).astype(float)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(reviews_train)
X_train = tokenizer.texts_to_sequences(reviews_train)
X_test = tokenizer.texts_to_sequences(reviews_test)
vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

print(reviews_train[2])
print(X_train[2])

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

print(X_train[0, :])

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset, 
                    validation_steps=30)
model.save('tomato1')
plot_graphs(history, accuracy)
