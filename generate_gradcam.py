import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image

# Configuration
TEST_IMAGES_DIR = r"C:\Users\LOQ\Desktop\test_images"
OUTPUT_DIR = "gradcam_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Image size expected by the model
IMG_SIZE = (224, 224)

# Grad-CAM functions
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """Generate class activation heatmap using Grad-CAM method."""
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

def _find_last_conv2d_layer_name(model):
    """Find the name of the last convolutional layer in the model."""
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            return layer.name
    return None

def load_model(model_name):
    """Load a pre-trained model by name."""
    model_path = os.path.join("results", "models", f"{model_name}_best.keras")
    if not os.path.exists(model_path):
        print(f"Model {model_name} not found at {model_path}")
        return None
    
    print(f"Loading model from: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"Successfully loaded {model_name}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def get_sample_images(num_images=5):
    """Get sample images from the test directory."""
    # Get all image files
    image_files = [os.path.join(TEST_IMAGES_DIR, f) for f in os.listdir(TEST_IMAGES_DIR) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Limit to num_images
    image_files = image_files[:num_images]
    
    if not image_files:
        print(f"No images found in {TEST_IMAGES_DIR}")
        return []
    
    print(f"Found {len(image_files)} images. Processing first {num_images}...")
    
    # Load and preprocess images
    images = []
    for img_path in image_files:
        try:
            # Load and preprocess image
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create batch dimension
            
            # Preprocess for the specific model (using EfficientNet preprocessing as default)
            img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
            images.append(img_array[0])  # Remove batch dimension
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    return np.array(images)

def generate_gradcam_for_model(model, model_name, images):
    """Generate Grad-CAM visualizations for a model."""
    # Get the last convolutional layer
    last_conv_layer_name = _find_last_conv2d_layer_name(model)
    if not last_conv_layer_name:
        print(f"Could not find a Conv2D layer in {model_name}")
        return []
    
    print(f"\nGenerating Grad-CAM for {model_name} using layer: {last_conv_layer_name}")
    
    # Create output directory
    output_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = []
    
    for i, img_array in enumerate(images):
        try:
            # Generate heatmap
            heatmap = make_gradcam_heatmap(
                np.expand_dims(img_array, axis=0),  # Add batch dimension
                model,
                last_conv_layer_name
            )
            
            # Convert to 0-255 range
            img_array_uint8 = ((img_array - img_array.min()) * (255 / (img_array.max() - img_array.min()))).astype('uint8')
            
            # Resize heatmap to the original image size
            heatmap = cv2.resize(heatmap, (img_array_uint8.shape[1], img_array_uint8.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            
            # Apply colormap to heatmap
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Superimpose the heatmap on original image
            superimposed_img = heatmap * 0.4 + img_array_uint8
            superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
            
            # Save the visualization
            output_path = os.path.join(output_dir, f"gradcam_{i}.png")
            cv2.imwrite(output_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
            
            # Also save the original image for comparison
            original_path = os.path.join(output_dir, f"original_{i}.png")
            cv2.imwrite(original_path, cv2.cvtColor(img_array_uint8, cv2.COLOR_RGB2BGR))
            
            saved_paths.append(output_path)
            print(f"  - Saved Grad-CAM visualization to {output_path}")
            
        except Exception as e:
            print(f"Error processing image {i}: {e}")
    
    return saved_paths

def main():
    # Get sample images
    print("Loading sample images...")
    sample_images = get_sample_images(num_images=5)
    if len(sample_images) == 0:
        print(f"No test images found in {TEST_IMAGES_DIR}. Please check the directory.")
        return
    
    print(f"Successfully loaded {len(sample_images)} images.")
    
    # List of available models
    model_names = ['EfficientNetB0', 'InceptionV3', 'ResNet50']
    
    # Generate Grad-CAM for each model
    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"PROCESSING MODEL: {model_name}")
        print(f"{'='*60}")
        
        # Load the model
        model = load_model(model_name)
        if model is None:
            print(f"Skipping {model_name} as it couldn't be loaded.")
            continue
            
        # Generate Grad-CAM visualizations
        generate_gradcam_for_model(model, model_name, sample_images)
    
    print("\n" + "="*60)
    print("GRAD-CAM GENERATION COMPLETE")
    print("="*60)
    print(f"All visualizations have been saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("For each model, you'll find both the original and Grad-CAM visualizations.")

if __name__ == "__main__":
    main()
