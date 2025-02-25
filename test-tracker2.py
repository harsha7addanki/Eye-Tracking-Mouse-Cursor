import tkinter as tk
from tkinter import Canvas
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, LayerNormalization, MultiHeadAttention
import numpy as np

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

# Register the custom objects
tf.keras.utils.get_custom_objects().update({
    'SpatialAttentionBlock': SpatialAttentionBlock,
    'ResidualBlock': ResidualBlock
})

# Now load the model with custom objects
model = load_model('best_model2.keras')

# Initialize the main window
root = tk.Tk()
root.title("Eye Tracking Mouse Cursor")
root.state('zoomed')

# Create a canvas
canvas = Canvas(root, width=600, height=600)
canvas.pack(expand=True, fill='both')

# Initialize the video capture
cap = cv2.VideoCapture(0)

def preprocess_image(frame, target_size=(128, 128)):
    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Apply CLAHE for better contrast
    lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[...,0] = clahe.apply(lab[...,0])
    frame_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Resize
    resized = cv2.resize(frame_rgb, target_size)
    
    # Normalize
    normalized = resized.astype('float32') / 255.0
    return normalized

def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

    # Preprocess the frame for the model
    processed_frame = preprocess_image(frame)
    input_data = np.expand_dims(processed_frame, axis=0)

    # Predict the gaze coordinates
    prediction = model.predict(input_data, verbose=0)
    
    # Extract x and y coordinates from the single prediction array
    x, y = prediction[0]
    
    # Scale the coordinates to the canvas size
    canvas_x = int(x * canvas.winfo_width())
    canvas_y = int(y * canvas.winfo_height())

    # Clear the canvas and draw the dot
    canvas.delete("all")
    canvas.create_oval(canvas_x - 5, canvas_y - 5, canvas_x + 5, canvas_y + 5, fill="red")

    # Schedule the next frame update
    root.after(10, update_frame)

# Start the frame update loop
update_frame()

# Start the Tkinter main loop
root.mainloop()

# Release the video capture when the window is closed
cap.release()
cv2.destroyAllWindows()