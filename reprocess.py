import os
import numpy as np
from PIL import Image, ImageOps

# Threshold value matching the ESP32's resize_96x96_to_32x32_and_threshold function
# Pixels >= 128 become white (255), pixels < 128 become black (0)
THRESHOLD = 128

def process_image(filepath):
    """Preprocess a raw 96x96 image to match ESP32's inference preprocessing exactly.
    
    The ESP32 uses pixel sampling (every 3rd pixel) to resize from 96x96 to 32x32,
    then applies a binary threshold. This function replicates that exact process
    on the laptop so training data matches what the model sees at inference time.
    
    Args:
        filepath: Path to raw 96x96 BMP image
    Returns:
        PIL Image: 32x32 binary (black/white) image
    """
    img = Image.open(filepath).convert("L")  # Open as grayscale
    arr = np.array(img)  # Convert to numpy array (96x96)
    
    # Sample every 3rd pixel in both dimensions: indices 0, 3, 6, ... 93
    # This produces a 32x32 array, matching ESP32's sampling method
    sampled = arr[::3, ::3]
    
    # Apply binary threshold at 128 - same as ESP32's threshold parameter
    thresholded = np.where(sampled >= THRESHOLD, 255, 0).astype(np.uint8)
    
    return Image.fromarray(thresholded)

def augment_image(img):
    """Generate augmented versions of a processed image to increase dataset size.
    
    Augmentations simulate natural variation in hand position and orientation.
    Each raw image produces 4 processed images (original + 3 augmentations).
    
    Args:
        img: PIL Image (32x32 binary)
    Returns:
        list of PIL Images
    """
    augmented = [img]                          # Original image
    augmented.append(ImageOps.mirror(img))     # Horizontal flip
    augmented.append(img.rotate(10))           # Slight clockwise rotation
    augmented.append(img.rotate(-10))          # Slight counter-clockwise rotation
    return augmented

counter = 0  # Global counter for unique output filenames

for cls in ["rock", "paper", "scissors"]:
    raw_folder = f"dataset/raw/{cls}"       # Input: raw captured images
    proc_folder = f"dataset/processed/{cls}"  # Output: preprocessed images
    os.makedirs(proc_folder, exist_ok=True)

    # Clear old processed images to avoid stale data from previous runs
    for f in os.listdir(proc_folder):
        os.remove(os.path.join(proc_folder, f))

    # Process each raw image and save all augmented versions
    for filename in sorted(os.listdir(raw_folder)):
        if filename.endswith(".bmp"):
            filepath = os.path.join(raw_folder, filename)
            processed = process_image(filepath)
            for aug in augment_image(processed):
                counter += 1
                aug.save(os.path.join(proc_folder, f"{cls}_{counter:04d}.bmp"))

    print(f"{cls}: {len(os.listdir(proc_folder))} images")

print(f"Total: {counter} images")
