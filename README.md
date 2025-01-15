# Fashion-MNIST
Fashion MNIST Classification Using CNN in Python and R
Overview

This project demonstrates the implementation of a Convolutional Neural Network (CNN) for classifying the Fashion MNIST dataset using both Python and R. The Fashion MNIST dataset contains grayscale images of clothing items, classified into 10 categories. The CNN is built with six layers to achieve high accuracy in image classification.

The project includes:

    Python implementation of the CNN.
    R implementation of the CNN.
    Steps to predict and evaluate the performance of the models on the test dataset.

Project Files

    fashion_mnist_cnn_python.py: Python script containing the CNN implementation.
    fashion_mnist_cnn_r.R: R script containing the CNN implementation.
    README.md: Instructions for understanding and running the project.
    requirements.txt: List of Python dependencies required for running the Python script.

Prerequisites
For Python

Ensure you have Python 3.8+ installed with the following libraries:

    tensorflow
    keras
    numpy
    matplotlib

Install these dependencies using the command:

pip install -r requirements.txt

For R

Ensure you have the following installed:

    R 4.0+
    Packages: keras, tensorflow, reticulate

Install R packages using:

install.packages(c("keras", "tensorflow", "reticulate"))

Configure TensorFlow in R:

library(keras)
install_keras()

Step-by-Step Instructions
1. Load the Fashion MNIST Dataset

Both Python and R scripts load the Fashion MNIST dataset. The dataset contains:

    Training set: 60,000 images.
    Test set: 10,000 images.

Python:

from tensorflow.keras.datasets import fashion_mnist

# Load dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

R:

library(keras)

# Load dataset
fashion_mnist <- dataset_fashion_mnist()
X_train <- fashion_mnist$train$x
y_train <- fashion_mnist$train$y
X_test <- fashion_mnist$test$x
y_test <- fashion_mnist$test$y

2. Preprocess the Data

Normalize the pixel values to be between 0 and 1, and reshape the data to include a channel dimension.

Python:

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

R:

# Normalize
X_train <- X_train / 255
X_test <- X_test / 255

# Reshape
X_train <- array_reshape(X_train, c(dim(X_train)[1], 28, 28, 1))
X_test <- array_reshape(X_test, c(dim(X_test)[1], 28, 28, 1))

3. Build the CNN Model

Define a CNN with six layers, including convolutional, pooling, and dense layers.

Python:

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

R:

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

4. Compile the Model

Use cross-entropy loss and the Adam optimizer.

Python:

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

R:

model %>% compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = 'accuracy'
)

5. Train the Model

Fit the model on the training dataset.

Python:

model.fit(X_train, y_train, epochs=10, validation_split=0.2, batch_size=32)

R:

model %>% fit(
  X_train, y_train,
  epochs = 10, validation_split = 0.2, batch_size = 32
)

6. Evaluate the Model

Test the model's performance on the test dataset.

Python:

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

R:

scores <- model %>% evaluate(X_test, y_test)
cat('Test Accuracy:', scores[[2]])

7. Make Predictions

Use the trained model to make predictions for two test images.

Python:

predictions = model.predict(X_test[:2])
print(predictions.argmax(axis=1))

R:

predictions <- model %>% predict(X_test[1:2, , , ])
apply(predictions, 1, which.max) - 1

Results

    Test Accuracy: Achieved ~90.4% accuracy on the test dataset.
    Predicted classes for the first two test images matched their true labels, demonstrating good performance.

How to Run the Code

    Clone the repository or unzip the project files.

    Install the required dependencies.

    Run the scripts:
        For Python:

        python fashion_mnist_cnn_python.py

        For R:
        Open fashion_mnist_cnn_r.R in RStudio and run the script line by line.

    Review the outputs, including the test accuracy and predictions.

Project Structure

/project_directory
|-- fashion_mnist_cnn_python.py   # Python script
|-- fashion_mnist_cnn_r.R         # R script
|-- requirements.txt              # Python dependencies
|-- README.md                     # Instructions and details
