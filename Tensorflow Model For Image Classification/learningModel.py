# Machine Learning Convolutional Neural Network Model

# Import the required libraries

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# import pandas as pd
 
# Load and Read the data 

def mainFunction():

    # Import pandas library
    import pandas as pd

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