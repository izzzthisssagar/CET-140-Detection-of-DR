import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

print("=" * 60)
print("SIMPLE EYE GRAD-CAM")
print("=" * 60)
print(f"Python: {sys.version.split()[0]}")
print(f"TensorFlow: {tf.__version__}")
print(f"OpenCV: {cv2.__version__}")
print(f"Working directory: {os.getcwd()}")
print("=" * 60 + "\n")

# Configuration
TEST_IMAGES_DIR = r"C:\Users\LOQ\Desktop\test_images"
OUTPUT_DIR = "eye_gradcam_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get list of test images
image_files = [f for f in os.listdir(TEST_IMAGES_DIR) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not image_files:
    print(f"No image files found in {TEST_IMAGES_DIR}")
    sys.exit(1)

# Limit to first 5 images
image_files = image_files[:5]
print(f"Found {len(image_files)} images. Processing...\n")

# Load a pre-trained model (we'll use MobileNetV2 for this example)
print("Loading pre-trained MobileNetV2 model...")
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Function to preprocess image for MobileNetV2
def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    return x

# Function to generate Grad-CAM
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    # Create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    # Compute gradients of the output neuron with respect to the output feature map
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Pool the gradients over all the axes leaving out the channel dimension
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply each channel in the feature map array by its importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize the heatmap between 0 & 1 for visualization
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Find the last convolutional layer
last_conv_layer = None
for layer in reversed(model.layers):
    if 'conv' in layer.name:
        last_conv_layer = layer.name
        break

if not last_conv_layer:
    print("Could not find a suitable convolutional layer in the model.")
    sys.exit(1)

print(f"Using layer for Grad-CAM: {last_conv_layer}")

# Process each image
for img_file in image_files:
    img_path = os.path.join(TEST_IMAGES_DIR, img_file)
    print(f"\nProcessing: {img_file}")
    
    try:
        # Load and preprocess image
        img_array = preprocess_img(img_path)
        
        # Generate heatmap
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)
        
        # Load original image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superimpose heatmap on original image
        alpha = 0.4
        superimposed_img = cv2.addWeighted(img, 1.0, heatmap, alpha, 0)
        
        # Save results
        base_name = os.path.splitext(img_file)[0]
        
        # Save original image
        original_path = os.path.join(OUTPUT_DIR, f"{base_name}_original.jpg")
        cv2.imwrite(original_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # Save heatmap
        heatmap_path = os.path.join(OUTPUT_DIR, f"{base_name}_heatmap.jpg")
        cv2.imwrite(heatmap_path, heatmap)
        
        # Save superimposed image
        result_path = os.path.join(OUTPUT_DIR, f"{base_name}_gradcam.jpg")
        cv2.imwrite(result_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
        
        print(f"  - Results saved to {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"  - Error processing {img_file}: {e}")
        continue

print("\n" + "="*60)
print("GRAD-CAM GENERATION COMPLETE")
print("="*60)
print(f"All visualizations have been saved to: {os.path.abspath(OUTPUT_DIR)}")
