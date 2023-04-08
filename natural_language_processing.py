import numpy as np
import pandas as pd
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.utils import pad_sequences
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Load the dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# Truncate and pad input sequences
max_words = 500
X_train = pad_sequences(X_train, maxlen=max_words)
X_test = pad_sequences(X_test, maxlen=max_words)

# One-hot encode the output labels
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


# Define the model
model = Sequential()
model.add(Embedding(10000, 64, input_length=max_words))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Train the model
history = model.fit(X_train, y_train, validation_split=0.1, epochs=5, batch_size=64)

# Evaluate the model

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Generate a confusion matrix and classification report
from sklearn.metrics import confusion_matrix, classification_report
y_pred = model.predict(X_test)
y_test = np.argmax(y_test, axis=1)
#cm = confusion_matrix(y_test, y_pred)
#print("Confusion Matrix:\n", cm)
#cr = classification_report(y_test, y_pred)
#print("Classification Report:\n", cr)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()