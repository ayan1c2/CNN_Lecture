import tensorflow as tf
from tensorflow import keras

# lspci | grep -i nvidia - get GPU Info
# nvidia-smi  # For NVIDIA to check GPU Drivers

# Check if TensorFlow detects a GPU
print("TensorFlow Version:", tf.__version__)
print("Keras Version:", tf.keras.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

# Print GPU details
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    print("Using GPU:", gpu_devices)
else:
    print("No GPU detected. Running on CPU.")

# Create a small Sequential model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),  # Input layer
    keras.layers.Dense(32, activation='relu'),  # Hidden layer
    keras.layers.Dense(1, activation='sigmoid')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()
