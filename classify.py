import gc
import array
import time
from camera import Camera, PixelFormat, FrameSize
from image_preprocessing import resize_96x96_to_32x32_and_threshold
from image_preprocessing import strip_bmp_header
import emlearn_cnn_fp32 as emlearn_cnn

# Path to the trained model file stored on the ESP32 filesystem
MODEL = 'prs_cnn.tmdl'

# Camera hardware pin configuration for the ESP32-S3 board
# These match the physical wiring of the OV2640 camera module
CAMERA_PARAMETERS = {
    "data_pins": [15, 17, 18, 16, 14, 12, 11, 48],  # 8-bit parallel data bus
    "vsync_pin": 38,    # Vertical sync signal
    "href_pin": 47,     # Horizontal reference signal
    "sda_pin": 40,      # I2C data for camera config
    "scl_pin": 39,      # I2C clock for camera config
    "pclk_pin": 13,     # Pixel clock
    "xclk_pin": 10,     # External clock input to camera
    "xclk_freq": 20000000,  # 20 MHz clock frequency
    "powerdown_pin": -1,    # Not used
    "reset_pin": -1,        # Not used
    "frame_size": FrameSize.R96X96,       # Capture at 96x96 pixels
    "pixel_format": PixelFormat.GRAYSCALE # Grayscale mode matches training data
}

def argmax(arr):
    """Return the index of the highest value in an array.
    Used to find which class has the highest probability."""
    max_val = arr[0]
    max_idx = 0
    for i in range(1, len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
            max_idx = i
    return max_idx

# Initialize and configure the camera
cam = Camera(**CAMERA_PARAMETERS)
cam.init()
cam.set_bmp_out(True)  # Output images in BMP format for easy parsing

# Load the trained CNN model from flash into RAM
print("Loading model...")
with open(MODEL, 'rb') as f:
    model_data = array.array('B', f.read())  # Read model as byte array
    gc.collect()  # Free memory before allocating model
    model = emlearn_cnn.new(model_data)       # Initialize CNN with model weights
    print("Model loaded!")

# Class labels matching the order used during training
classes = ['rock', 'paper', 'scissors']

# Output buffer for the 3 class probabilities from the CNN
probabilities = array.array('f', [0.0] * 3)

# Smoothing and stability parameters
CONFIDENCE_THRESHOLD = 0.90  # Minimum confidence to consider a prediction valid
BUFFER_SIZE = 5               # Number of consecutive consistent frames required
LOCK_SECONDS = 3              # Seconds to lock prediction after a confident detection

recent = []        # Rolling buffer of recent predictions
locked_until = 0   # Timestamp until which predictions are locked
hand_was_away = True  # Tracks whether hand has been removed between gestures

print("Ready! Show a gesture... (Ctrl+C to stop)")

while True:
    # Step 1: Capture a 96x96 grayscale BMP image from the camera
    img = cam.capture()

    # Step 2: Resize to 32x32 and apply binary threshold at 128
    # This matches exactly how training images were preprocessed
    small = resize_96x96_to_32x32_and_threshold(img, 128)

    # Step 3: Strip the BMP file header to get raw 1024-byte pixel array
    raw = strip_bmp_header(small)

    # Step 4: Convert to array.array('B') as required by emlearn_cnn
    input_data = array.array('B', raw)

    # Step 5: Run CNN inference - fills probabilities with scores for each class
    model.run(input_data, probabilities)

    now = time.time()
    confidence = max(probabilities)
    prediction = classes[argmax(probabilities)]

    # Calculate margin between top two predictions
    # A small margin means the model is uncertain - likely no clear gesture
    second_best = sorted(probabilities)[-2]
    margin = confidence - second_best

    # If predictions are too similar, treat as no gesture (hand away)
    if margin < 0.10:
        hand_was_away = True
        recent = []
        continue

    # Skip if still within the lock period after a previous detection
    if now < locked_until:
        continue

    # Require hand to have been removed before accepting a new gesture
    # This prevents the same gesture from being printed multiple times
    if not hand_was_away:
        continue

    # Add to rolling buffer if confident enough
    if confidence >= CONFIDENCE_THRESHOLD:
        recent.append(prediction)
    else:
        recent = []  # Reset buffer if confidence drops

    # Only print if the last BUFFER_SIZE frames all agree on the same gesture
    if len(recent) >= BUFFER_SIZE and len(set(recent)) == 1:
        print(f">>> {prediction.upper()}  ({confidence*100:.0f}%)")
        locked_until = now + LOCK_SECONDS  # Lock for 3 seconds
        hand_was_away = False              # Require hand removal before next detection
        recent = []                        # Reset buffer
