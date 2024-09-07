import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv1D, MaxPooling1D, BatchNormalization, Flatten
import h5py
import numpy as np

# Load the dataset
def load_h5_data(filename):
    with h5py.File(filename, 'r') as f:
        data = f['data'][:]
        labels = f['labels'][:]
    return data, labels

# Define the PointNet model
def build_pointnet(input_shape):
    inputs = Input(shape=input_shape)
    
    # First conv layer
    x = Conv1D(64, 1, activation='relu', input_shape=input_shape)(inputs)
    x = BatchNormalization()(x)
    
    # Second conv layer
    x = Conv1D(128, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Third conv layer
    x = Conv1D(1024, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Max pooling
    x = MaxPooling1D(pool_size=input_shape[0])(x)
    x = Flatten()(x)
    
    # Fully connected layers
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    
    # Output layer (for regression task predicting bounding box vertices)
    outputs = Dense(24, activation='linear')(x)  # 24D output for 8 bounding box vertices (3 coords each)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Load data and labels
data, labels = load_h5_data('pointnet_data.h5')
print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")

# Process the labels
labels = np.mean(labels, axis=1)  # Averaging the two sets of 24 values per sample
print(f"Processed Labels shape: {labels.shape}")

# Reshape the labels to match the model's output shape
assert data.shape[0] == labels.shape[0], "Mismatch in number of samples between data and labels."

# Build the model
input_shape = (data.shape[1], data.shape[2])  # (num_points, 6) - XYZ + RGB
model = build_pointnet(input_shape)

# Compile the model with a regression loss function (mean squared error)
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(data, labels, epochs=50, batch_size=16)

# Save the trained model
model.save('pointnet_model.h5')
