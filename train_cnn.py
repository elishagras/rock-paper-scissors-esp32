import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

def load_bmp_raw(filepath):
    """Load pixel data from a 32x32 8-bit grayscale BMP file.
    
    BMP file structure:
    - 14 bytes: BMP file header
    - 40 bytes: DIB header
    - 256 * 4 = 1024 bytes: Color palette (256 colors x 4 bytes each)
    - 1024 bytes: Pixel data (32 x 32 pixels, 1 byte each)
    
    We skip directly to pixel data at offset 1078.
    """
    with open(filepath, 'rb') as f:
        data = f.read()
    pixel_start = 14 + 40 + 256 * 4  # = 1078
    pixels = data[pixel_start:pixel_start + 1024]  # 32x32 = 1024 pixels
    return np.frombuffer(pixels, dtype=np.uint8).reshape(32, 32)

def load_dataset(base_path):
    """Load all processed BMP images from the dataset directory.
    
    Expects folder structure:
      base_path/rock/      - rock gesture images
      base_path/paper/     - paper gesture images
      base_path/scissors/  - scissors gesture images
    
    Returns:
      images: numpy array of shape (N, 32, 32)
      labels: numpy array of shape (N,) with values 0=rock, 1=paper, 2=scissors
    """
    images = []
    labels = []
    classes = ["rock", "paper", "scissors"]
    
    for label, cls in enumerate(classes):
        folder = os.path.join(base_path, cls)
        for filename in os.listdir(folder):
            if filename.endswith(".bmp"):
                path = os.path.join(folder, filename)
                img = load_bmp_raw(path)
                images.append(img)
                labels.append(label)
    
    return np.array(images), np.array(labels)

print("Loading images...")
X, y = load_dataset("dataset/processed")
print(f"Loaded {len(X)} images")
print(f"Rock: {np.sum(y==0)}, Paper: {np.sum(y==1)}, Scissors: {np.sum(y==2)}")

# Normalize pixel values from [0, 255] to [0.0, 1.0]
# Since images are binary (0 or 255), this maps to exactly 0.0 or 1.0
X = X.astype("float32") / 255.0

# Reshape to (N, height, width, channels) - Keras expects 4D input
X = X.reshape(-1, 32, 32, 1)

# One-hot encode labels: e.g. rock=0 becomes [1, 0, 0]
y = to_categorical(y, 3)

# Split into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training: {len(X_train)}, Testing: {len(X_test)}")

# Build CNN model
# Uses strided convolutions instead of MaxPooling for ESP32 compatibility
# (emlearn tmdl format does not support MaxPooling layers)
model = Sequential([
    # First conv layer: 8 filters, 3x3 kernel, stride 2 to downsample 32x32 -> 15x15
    Conv2D(8, (3, 3), strides=(2, 2), activation="relu", input_shape=(32, 32, 1)),
    # Second conv layer: 16 filters, 3x3 kernel, stride 2 to downsample 15x15 -> 7x7
    Conv2D(16, (3, 3), strides=(2, 2), activation="relu"),
    # Flatten 7x7x16 = 784 features into a 1D vector
    Flatten(),
    # Fully connected layer to learn high-level combinations of features
    Dense(32, activation="relu"),
    # Output layer: 3 classes (rock, paper, scissors) with softmax probabilities
    Dense(3, activation="softmax")
])

# Adam optimizer with categorical crossentropy for multi-class classification
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

print("\nTraining...")
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate final performance on held-out test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {accuracy * 100:.1f}%")

# Save model in Keras HDF5 format for subsequent conversion
model.save("rps_model.h5")
print("Model saved as rps_model.h5")
