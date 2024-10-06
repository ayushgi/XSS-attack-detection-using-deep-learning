import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('XSS-attack-detection-using-deep-learning\XSS_dataset_mixed.csv', encoding='utf-8-sig')

# Extract features and labels
sentences = df['Sentence'].values
labels = df['Label'].values

# Set maximum vocabulary size and maximum sequence length
vocab_size = 10000
max_length = 400
embedding_dim = 128

# Initialize the tokenizer and fit on the sentences
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

# Convert sentences to sequences
sequences = tokenizer.texts_to_sequences(sentences)

# Pad the sequences to ensure uniform length
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# Split the dataset into training and testing sets
trainX, testX, trainY, testY = train_test_split(padded_sequences, labels, test_size=0.4, random_state=42)

# Build the GRU model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    GRU(128, return_sequences=True),
    GRU(64),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(testX, testY)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Plot the training and validation accuracy and loss
plt.figure(figsize=(14, 6))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], 'g', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], 'g', label='Training Loss')
plt.plot(history.history['val_loss'], 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

# Make predictions and evaluate using a confusion matrix
predictions = (model.predict(testX) > 0.5).astype("int32")
cm = confusion_matrix(testY, predictions)

plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-XSS', 'XSS'], yticklabels=['Non-XSS', 'XSS'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
# Save the model
model.save('xss_detection_model.h5')

