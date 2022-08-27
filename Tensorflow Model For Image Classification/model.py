# Machine Learning Convolutional Neural Network Model

# Import the required libraries

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

# Load and Read the data 

source_data = pd.read_csv('data/submission_file.csv')
print(source_data.info())