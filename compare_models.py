import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, LayerNormalization, MultiHeadAttention
import matplotlib.pyplot as plt
import pandas as pd

@tf.keras.utils.register_keras_serializable(package="eyetrack")
class SpatialAttentionBlock(Layer):
    def __init__(self, num_heads=4, key_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attention = None
        self.layernorm = LayerNormalization(epsilon=1e-6)
    
    def build(self, input_shape):
        _, h, w, c = input_shape
        self.key_dim = self.key_dim or c // self.num_heads
        self.attention = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            value_dim=self.key_dim,
            use_bias=True,
            dropout=0.1
        )
        self.built = True
    
    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        h = tf.shape(inputs)[1]
        w = tf.shape(inputs)[2]
        c = tf.shape(inputs)[3]
        
        x_seq = tf.reshape(inputs, (batch_size, h * w, c))
        x_norm = self.layernorm(x_seq)
        attention_output = self.attention(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            training=training
        )
        x_seq = x_seq + attention_output
        return tf.reshape(x_seq, (batch_size, h, w, c))
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package="eyetrack")
class ResidualBlock(Layer):
    def __init__(self, filters, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv1 = None
        self.bn1 = None
        self.act1 = None
        self.conv2 = None
        self.bn2 = None
        self.attention = None
        self.shortcut_conv = None
        self.shortcut_bn = None
        self.add = None
        self.act_out = None
    
    def build(self, input_shape):
        self.conv1 = tf.keras.layers.Conv2D(self.filters, self.kernel_size, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.LeakyReLU(negative_slope=0.1)
        
        self.conv2 = tf.keras.layers.Conv2D(self.filters, self.kernel_size, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        
        self.attention = SpatialAttentionBlock(
            num_heads=4,
            key_dim=self.filters // 4
        )
        
        if input_shape[-1] != self.filters:
            self.shortcut_conv = tf.keras.layers.Conv2D(self.filters, 1, padding='same')
            self.shortcut_bn = tf.keras.layers.BatchNormalization()
        else:
            self.shortcut_conv = None
            self.shortcut_bn = None
        
        self.add = tf.keras.layers.Add()
        self.act_out = tf.keras.layers.LeakyReLU(negative_slope=0.1)
        self.built = True
    
    def call(self, inputs, training=None):
        shortcut = inputs
        
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        x = self.attention(x, training=training)
        
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)
            shortcut = self.shortcut_bn(shortcut, training=training)
        
        x = self.add([shortcut, x])
        return self.act_out(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def preprocess_image(image_path, target_size=(128, 128)):
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply CLAHE for better contrast
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[...,0] = clahe.apply(lab[...,0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Resize
    img = cv2.resize(img, target_size)
    return img

# Register the custom objects
tf.keras.utils.get_custom_objects().update({
    'SpatialAttentionBlock': SpatialAttentionBlock,
    'ResidualBlock': ResidualBlock
})

# Load both models
model1 = load_model('best_model.keras')
model2 = load_model('best_model2.keras')

# Load labels to get actual coordinates
labels_df = pd.read_csv('./test_labels.csv')

# Load and test multiple random images
data_dir = 'test'
# Get list of all jpg files that have corresponding labels
available_images = []
for f in os.listdir(data_dir):
    if f.endswith('.jpg'):
        image_id = f.split('.')[0]
        if len(labels_df[labels_df['image'] == image_id]) > 0:
            available_images.append(f)

if len(available_images) < 10:
    num_images = len(available_images)
    print(f"\nWarning: Only {num_images} labeled images available. Will test all of them.")
else:
    num_images = 10

# Randomly select images from available ones
selected_images = np.random.choice(available_images, size=min(10, len(available_images)), replace=False)

# Lists to store errors for averaging for both models
model1_x_errors = []
model1_y_errors = []
model1_euclidean_errors = []
model2_x_errors = []
model2_y_errors = []
model2_euclidean_errors = []

# Create a figure for all images
plt.figure(figsize=(20, 8 * ((num_images + 1) // 2)))

for idx, image_name in enumerate(selected_images):
    test_image = os.path.join(data_dir, image_name)
    print(f"\nTesting image {idx + 1}/{num_images}: {image_name}")

    # Get actual coordinates from labels
    actual_coords = labels_df[labels_df['image'] == image_name.split('.')[0]].iloc[0]
    actual_x = float(actual_coords.iloc[1])
    actual_y = float(actual_coords.iloc[2])

    # Preprocess the image
    processed_image = preprocess_image(test_image)
    normalized_image = processed_image.astype('float32') / 255.0
    input_data = np.expand_dims(normalized_image, axis=0)

    # Get predictions from both models
    prediction1 = model1.predict(input_data, verbose=0)
    x1, y1 = prediction1  # Model 1 output format
    
    prediction2 = model2.predict(input_data, verbose=0)
    x2, y2 = prediction2[0]  # Model 2 output format

    # Calculate errors for model 1
    x1_error = abs(x1 - actual_x)
    y1_error = abs(y1 - actual_y)
    euclidean1_error = np.sqrt((x1 - actual_x)**2 + (y1 - actual_y)**2)
    
    model1_x_errors.append(x1_error)
    model1_y_errors.append(y1_error)
    model1_euclidean_errors.append(euclidean1_error)

    # Calculate errors for model 2
    x2_error = abs(x2 - actual_x)
    y2_error = abs(y2 - actual_y)
    euclidean2_error = np.sqrt((x2 - actual_x)**2 + (y2 - actual_y)**2)
    
    model2_x_errors.append(x2_error)
    model2_y_errors.append(y2_error)
    model2_euclidean_errors.append(euclidean2_error)

    print(f"\nModel 1 Results:")
    print("Predicted coordinates: x=", x1, "y=", y1)
    print("Error: x=", x1_error, "y=", y1_error, "euclidean=", euclidean1_error)
    
    print(f"\nModel 2 Results:")
    print("Predicted coordinates: x=", x2, "y=", y2)
    print("Error: x=", x2_error, "y=", y2_error, "euclidean=", euclidean2_error)

    # Plot in a 5x6 grid (10 sets of 3 plots)
    # Original image with both predictions
    plt.subplot(5, 6, idx*3 + 1)
    plt.imshow(processed_image)
    plt.plot(x1 * processed_image.shape[1], y1 * processed_image.shape[0], 'r+', markersize=20, label='Model 1')
    plt.plot(x2 * processed_image.shape[1], y2 * processed_image.shape[0], 'b+', markersize=20, label='Model 2')
    plt.plot(actual_x * processed_image.shape[1], actual_y * processed_image.shape[0], 'g+', markersize=20, label='Actual')
    plt.title(f'Image {idx + 1}')
    if idx == 0:  # Only show legend for first plot to save space
        plt.legend()

    # Model 1 coordinate visualization
    plt.subplot(5, 6, idx*3 + 2)
    plt.plot(x1, y1, 'ro', markersize=10, label='Model 1')
    plt.plot(actual_x, actual_y, 'go', markersize=10, label='Actual')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.title(f'Model 1 Coords')
    if idx == 0:
        plt.legend()

    # Model 2 coordinate visualization
    plt.subplot(5, 6, idx*3 + 3)
    plt.plot(x2, y2, 'bo', markersize=10, label='Model 2')
    plt.plot(actual_x, actual_y, 'go', markersize=10, label='Actual')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.title(f'Model 2 Coords')
    if idx == 0:
        plt.legend()

# Calculate and print average errors for both models
print("\nModel 1 Average Errors:")
print("X Error:", np.mean(model1_x_errors))
print("Y Error:", np.mean(model1_y_errors))
print("Euclidean Error:", np.mean(model1_euclidean_errors))

print("\nModel 2 Average Errors:")
print("X Error:", np.mean(model2_x_errors))
print("Y Error:", np.mean(model2_y_errors))
print("Euclidean Error:", np.mean(model2_euclidean_errors))

# Print model comparison
print("\nModel Comparison (Model 2 vs Model 1):")
x_improvement = (np.mean(model1_x_errors) - np.mean(model2_x_errors)) / np.mean(model1_x_errors) * 100
y_improvement = (np.mean(model1_y_errors) - np.mean(model2_y_errors)) / np.mean(model1_y_errors) * 100
euclidean_improvement = (np.mean(model1_euclidean_errors) - np.mean(model2_euclidean_errors)) / np.mean(model1_euclidean_errors) * 100

print("X Error Improvement:", x_improvement, "%")
print("Y Error Improvement:", y_improvement, "%")
print("Euclidean Error Improvement:", euclidean_improvement, "%")

plt.tight_layout()
plt.show() 