# Machine Learning Convolutional Neural Network Model

# Import the required libraries

import numpy as np

import matplotlib.pyplot as plt

import tensorflow
import pandas as pd
# Load and Read the data 

def trainFunction():

    # Read train data source
    source_data = pd.read_csv('data/train.csv')

    # Print data source information
    print(source_data.info())

    # Import train_test_split function from sklearn libraries
    from sklearn.model_selection import train_test_split

    train_data, validation_data = train_test_split(source_data, test_size=0.25)

    # To check if the result is true
    print(train_data.info())
    print("-----")
    print(validation_data.info())

    # Erase the "labels" column
    train_data_label = train_data.pop('label')
    validation_data_label = validation_data.pop('label')

    # Generate tensorflow data object from the data variables given
    train_data_tensorflow = tensorflow.data.Dataset.from_tensor_slices((train_data.values,train_data_label.values))
    validation_data_tensorflow = tensorflow.data.Dataset.from_tensor_slices((validation_data.values,validation_data_label.values))

    # Reshape, scale and chnage the tensorflow object data type for the model

    def image_reshape_and_scaler(image_data, label_data):
        image_data = tensorflow.reshape(image_data, [28,28,1])
        image_data = tensorflow.cast(image_data, tensorflow.float32) / 255.

        return image_data, label_data

    train_data_tensorflow = train_data_tensorflow.map(image_reshape_and_scaler, num_parallel_calls=tensorflow.data.experimental.AUTOTUNE)

    validation_data_tensorflow = validation_data_tensorflow.map(image_reshape_and_scaler, num_parallel_calls=tensorflow.data.experimental.AUTOTUNE)

    # Create a function to shuffle the data, prefetch the data and create a batch threshold for the memory not to overload

    def shuffle_batch_prefetch(data_tensorflow):
        data_tensorflow = data_tensorflow.shuffle(100)
        data_tensorflow = data_tensorflow.batch(32)
        data_tensorflow = data_tensorflow.prefetch(tensorflow.data.experimental.AUTOTUNE)

        return data_tensorflow

    train_data_tensorflow = shuffle_batch_prefetch(train_data_tensorflow)

    validation_data_tensorflow = shuffle_batch_prefetch(validation_data_tensorflow)

    print(train_data_tensorflow)

    train_data_log = model.fit(train_data_tensorflow, validation_data=validation_data_tensorflow, epochs=30, callbacks=callbacks_model)

# Create the neural network model and train the model

model = tensorflow.keras.Sequential()

model.add(tensorflow.keras.layers.Conv2D(6, (5, 5), activation='relu', padding='same', input_shape=(28,28,1)))
model.add(tensorflow.keras.layers.MaxPooling2D(2,2))

model.add(tensorflow.keras.layers.Conv2D(16, (5, 5), activation='relu', padding='valid'))
model.add(tensorflow.keras.layers.MaxPooling2D(2,2))

model.add(tensorflow.keras.layers.Flatten())
model.add(tensorflow.keras.layers.Dense(120, activation='relu'))
model.add(tensorflow.keras.layers.Dense(84, activation='relu'))
model.add(tensorflow.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create the code statement to define the early stop and Learning Rate Decay to prevent from continuing an unnecessary process

callbacks_model = [
    tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=2, verbose=1),
    tensorflow.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1),
]



def testFunction(data_src):
    # Test the train model with the test datasets

    import numpy

    #source_test_data = pd.read_csv('data/test.csv')
    source_test_data = data_src
    source_test_data_tersorflow = tensorflow.data.Dataset.from_tensor_slices(
        (
            [
                source_test_data.to_numpy().reshape(len(source_test_data),28,28,1)
            ]
        )
    )
    print(source_test_data_tersorflow)

    prediction_data = model.predict(source_test_data_tersorflow)
    prediction_data = numpy.argmax(prediction_data, axis=1)

    # Generate the result to a csv file

    predictions_result = pd.DataFrame(
    data={'Label': prediction_data},
    index=pd.RangeIndex(start=1, stop=28001)
    )

    predictions_result.index = predictions_result.index.rename('Image Identification')

    predictions_result.to_csv('Result_of_Prediction.csv')