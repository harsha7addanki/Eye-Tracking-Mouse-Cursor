import os
import pandas as pd
from PIL import Image
import numpy as np
import tensorflow as tf

# Define paths
data_dir = 'data'
labels_file = './labels.csv'

# Load labels
labels_df = pd.read_csv(labels_file)

# Initialize lists to hold data
images = []
x_coords = []
y_coords = []

# Iterate through the labels and load corresponding images
for index, row in labels_df.iterrows():
    image_path = os.path.join(data_dir, row['image'])
    if os.path.exists(image_path):
        image = Image.open(image_path)
        images.append(np.array(image))
        x_coords.append(row['x'])
        y_coords.append(row['y'])

# Convert lists to numpy arrays
images = np.array(images)
x_coords = np.array(x_coords)
y_coords = np.array(y_coords)

# Combine into one dataset
dataset = {
    'images': images,
    'x_coords': x_coords,
    'y_coords': y_coords
}

# Create a TensorFlow dataset
tf_dataset = tf.data.Dataset.from_tensor_slices((images, {'x': x_coords, 'y': y_coords}))

# Example usage
print(f"Loaded {len(images)} images with corresponding coordinates.")

# Return the TensorFlow dataset
def get_dataset():
    return tf_dataset

print(tf_dataset.take(1))