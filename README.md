# MNIST Digit Classification with Neural Network

This project is a simple implementation of a neural network for classifying handwritten digits from the MNIST dataset using TensorFlow and Keras.

## Dataset

The MNIST dataset consists of 28x28 pixel grayscale images of handwritten digits (0 through 9). Each image is associated with a label indicating the digit it represents.

## Getting Started

1. **Load and Preprocess Data:**
   - Load the MNIST dataset using `keras.datasets.mnist.load_data()`.
   - Preprocess the input data by scaling pixel values from 0-255 to a range of 0-1.

    ```python
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    ```

2. **Build and Compile the Model:**
   - Create a sequential model with a Flatten layer for input and Dense layers with ReLU activation.
   - Use softmax activation in the output layer for multi-class classification.

    ```python
    from tensorflow.keras import Sequential
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    ```

3. **Train the Model:**
   - Fit the model to the training data for a specified number of epochs.

    ```python
    history = model.fit(X_train, y_train, epochs=20, validation_split=0.2)
    ```

4. **Evaluate Model Accuracy:**
   - Use the trained model to predict on the test set and calculate accuracy.

    ```python
    y_prob = model.predict(X_test)
    y_pred = y_prob.argmax(axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    ```

5. **Visualize Training History:**
   - Plot the training and validation loss and accuracy over epochs.

    ```python
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.show()
    ```

6. **Make Predictions:**
   - Visualize predictions by displaying an image from the test set and checking the model's prediction.

    ```python
    plt.imshow(X_test[0])
    plt.show()

    prediction = model.predict(X_test[0].reshape(1, 28, 28)).argmax(axis=1)
    print(f"Model Prediction: {prediction[0]}")
    ```

## Conclusion

This project demonstrates the process of building, training, and evaluating a neural network for digit classification using the MNIST dataset. The model achieves a high accuracy, showcasing the effectiveness of neural networks in image classification tasks. Feel free to explore and customize the code for further experimentation!
