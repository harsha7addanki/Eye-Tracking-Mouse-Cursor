import os
import pandas as pd
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization,
    Add, GlobalAveragePooling2D, LeakyReLU, MultiHeadAttention, LayerNormalization, Layer
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import cv2

# Define paths
data_dir = 'data'
labels_file = './labels.csv'

# Load labels
labels_df = pd.read_csv(labels_file)

# Initialize lists to hold data
images = []
x_coords = []
y_coords = []

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

# Iterate through the labels and load corresponding images with preprocessing
for index, row in labels_df.iterrows():
    image_path = os.path.join(data_dir, row['image'] + ".jpg")
    if os.path.exists(image_path):
        image = preprocess_image(image_path)
        images.append(image)
        # Use proper pandas indexing
        x_coords.append(np.array(row.iloc[1]))
        y_coords.append(np.array(row.iloc[2]))

# Convert to numpy arrays and normalize
images = np.array(images).astype('float32') / 255.0
x_coords = np.array(x_coords)
y_coords = np.array(y_coords)

# Create dataset without augmentation
tf_dataset = tf.data.Dataset.from_tensor_slices((images, (x_coords, y_coords)))

# Split dataset into train and validation
dataset_size = len(images)
train_size = int(0.8 * dataset_size)
train_dataset = tf_dataset.take(train_size).shuffle(1000).batch(32)
val_dataset = tf_dataset.skip(train_size).batch(32)

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
        
        # Reshape to sequence
        x_seq = tf.reshape(inputs, (batch_size, h * w, c))
        
        # Apply layer normalization
        x_norm = self.layernorm(x_seq)
        
        # Self-attention
        attention_output = self.attention(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            training=training
        )
        
        # Residual connection
        x_seq = x_seq + attention_output
        
        # Reshape back to original format
        return tf.reshape(x_seq, (batch_size, h, w, c))
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
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
        self.conv1 = Conv2D(self.filters, self.kernel_size, padding='same')
        self.bn1 = BatchNormalization()
        self.act1 = LeakyReLU(negative_slope=0.1)
        
        self.conv2 = Conv2D(self.filters, self.kernel_size, padding='same')
        self.bn2 = BatchNormalization()
        
        # Initialize attention with specific key_dim
        self.attention = SpatialAttentionBlock(
            num_heads=4,
            key_dim=self.filters // 4  # Divide by number of heads
        )
        
        if input_shape[-1] != self.filters:
            self.shortcut_conv = Conv2D(self.filters, 1, padding='same')
            self.shortcut_bn = BatchNormalization()
        else:
            self.shortcut_conv = None
            self.shortcut_bn = None
        
        self.add = Add()
        self.act_out = LeakyReLU(negative_slope=0.1)
        self.built = True
    
    def call(self, inputs, training=None):
        shortcut = inputs
        
        # First conv block
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Attention mechanism
        x = self.attention(x, training=training)
        
        # Adjust shortcut if needed
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)
            shortcut = self.shortcut_bn(shortcut, training=training)
        
        # Add and activate
        x = self.add([shortcut, x])
        return self.act_out(x)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.filters)
    
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

# Add assertions to verify registration
assert tf.keras.utils.get_registered_object('eyetrack>SpatialAttentionBlock') == SpatialAttentionBlock
assert tf.keras.utils.get_registered_object('eyetrack>ResidualBlock') == ResidualBlock
assert tf.keras.utils.get_registered_name(SpatialAttentionBlock) == 'eyetrack>SpatialAttentionBlock'
assert tf.keras.utils.get_registered_name(ResidualBlock) == 'eyetrack>ResidualBlock'

# Model architecture
def create_model(input_shape=(128, 128, 3)):
    image_input = Input(shape=input_shape)
    
    # Initial convolution
    x = Conv2D(64, (7, 7), strides=2, padding='same')(image_input)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
    
    # Residual blocks with increasing filters
    x = ResidualBlock(64)(x)
    x = ResidualBlock(128)(x)
    x = ResidualBlock(256)(x)
    x = ResidualBlock(512)(x)
    
    # Global features
    x = GlobalAveragePooling2D()(x)
    
    # Shared dense layers
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = Dropout(0.3)(x)
    
    # Branch 1 - maintain original dimension
    branch1 = Dense(512)(x)
    branch1 = BatchNormalization()(branch1)
    branch1 = LeakyReLU(negative_slope=0.1)(branch1)
    branch1 = Dropout(0.3)(branch1)
    
    # Branch 2 - project back to same dimension
    branch2 = Dense(256)(x)
    branch2 = BatchNormalization()(branch2)
    branch2 = LeakyReLU(negative_slope=0.1)(branch2)
    branch2 = Dropout(0.3)(branch2)
    branch2 = Dense(512)(branch2)  # Project back to 512 dimensions
    branch2 = BatchNormalization()(branch2)
    branch2 = LeakyReLU(negative_slope=0.1)(branch2)
    
    # Add the branches (now they have compatible shapes)
    x = Add()([branch1, branch2])
    
    # Separate paths for x and y coordinates
    # X coordinate path
    x_path = Dense(256)(x)
    x_path = BatchNormalization()(x_path)
    x_path = LeakyReLU(negative_slope=0.1)(x_path)
    x_path = Dropout(0.2)(x_path)
    x_path = Dense(128)(x_path)
    x_path = BatchNormalization()(x_path)
    x_path = LeakyReLU(negative_slope=0.1)(x_path)
    x_path = Dense(64)(x_path)
    x_coord = Dense(1, name='x_coord')(x_path)
    
    # Y coordinate path
    y_path = Dense(256)(x)
    y_path = BatchNormalization()(y_path)
    y_path = LeakyReLU(negative_slope=0.1)(y_path)
    y_path = Dropout(0.2)(y_path)
    y_path = Dense(128)(y_path)
    y_path = BatchNormalization()(y_path)
    y_path = LeakyReLU(negative_slope=0.1)(y_path)
    y_path = Dense(64)(y_path)
    y_coord = Dense(1, name='y_coord')(y_path)
    
    return Model(inputs=image_input, outputs=[x_coord, y_coord])

# Create and compile model
model = create_model()

# Modified optimizer configuration
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=1e-4,  # Lower initial learning rate
    weight_decay=1e-5    # Reduce weight decay
)

model.compile(
    optimizer=optimizer,
    loss={
        'x_coord': 'mse',  # Changed from huber to MSE
        'y_coord': 'mse'
    },
    loss_weights={
        'x_coord': 1.0,
        'y_coord': 1.0
    },
    metrics={
        'x_coord': ['mae', 'mse'],
        'y_coord': ['mae', 'mse']
    }
)

# Modified callbacks for more aggressive learning rate adjustment
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,      # Reduced patience
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,      # More aggressive reduction
        patience=10,      # Reduced patience
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        'best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# Modified training parameters
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=200, 
    callbacks=callbacks,
    verbose=1,
    shuffle=True        # Ensure shuffling
)
# Save the final model
model.save('final_model.keras')