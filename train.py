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
    image_path = os.path.join(data_dir, row['image'] + ".jpg")
    if os.path.exists(image_path):
        image = Image.open(image_path)
        images.append(np.array(image))
        x_coords.append(np.array(row[1]))
        y_coords.append(np.array(row[2]))

# force the images to be 128x128x3
for i in range(len(images)):
    images[i] = np.resize(images[i], (128, 128, 3))

# Convert lists to numpy arrays
x_coords = np.array(x_coords)
y_coords = np.array(y_coords)

# Combine into one dataset
dataset = {
    'images': images,
    'x_coords': x_coords,
    'y_coords': y_coords
}

print(x_coords)

# Create a TensorFlow dataset
tf_dataset = tf.data.Dataset.from_tensor_slices((np.array(images), (x_coords, y_coords)))

# Example usage
print(f"Loaded {len(images)} images with corresponding coordinates.")

# make a keras model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

image_input = Input(shape=(128, 128, 3))  # Adjust the input shape as needed

x = Conv2D(32, (3, 3), activation='relu')(image_input)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
y1 = Dense(1)(x)  # Output layer for x and y coordinates
y2 = Dense(1)(x)  # Output layer for z coordinate (if needed)

model = Model(inputs=image_input, outputs=[y1, y2])
# Compile the model with appropriate loss functions and optimizers
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mae'])
# Train the model
model.fit(tf_dataset.batch(2), epochs=100)  # Adjust batch size and epochs as needed

# Save the model
model.save('model.keras')