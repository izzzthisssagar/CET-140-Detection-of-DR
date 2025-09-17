"""
Utility script to analyze RGB channel statistics of training images.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm

def get_image_paths(directory):
    """Get all image file paths from the given directory and its subdirectories."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_paths = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths

def calculate_rgb_stats(image_paths, sample_size=None):
    """Calculate RGB channel statistics from a list of image paths."""
    if sample_size and sample_size < len(image_paths):
        image_paths = np.random.choice(image_paths, sample_size, replace=False)
    
    all_red, all_green, all_blue = [], [], []
    
    print(f"Analyzing {len(image_paths)} images...")
    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            
            all_red.extend(img_array[:, :, 0].flatten())
            all_green.extend(img_array[:, :, 1].flatten())
            all_blue.extend(img_array[:, :, 2].flatten())
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Convert to numpy arrays
    channels = {
        'red': np.array(all_red),
        'green': np.array(all_green),
        'blue': np.array(all_blue)
    }
    
    # Calculate statistics
    stats = {}
    for color, values in channels.items():
        if len(values) > 0:
            stats[color] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
    
    return stats, channels

def plot_channel_distributions(channels, save_path=None):
    """Plot the distribution of pixel values for each channel."""
    plt.figure(figsize=(12, 6))
    colors = ['red', 'green', 'blue']
    
    for i, (color, values) in enumerate(channels.items()):
        if len(values) > 0:
            plt.hist(values, bins=100, alpha=0.5, color=colors[i], 
                   label=f'{color.capitalize()} (μ={np.mean(values):.3f}, σ={np.std(values):.3f})')
    
    plt.title('RGB Channel Distributions')
    plt.xlabel('Pixel Value (Normalized)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to {save_path}")
    
    plt.close()

def main():
    # Define paths
    train_dir = os.path.join('data', 'train_images')
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image paths
    print(f"Searching for images in {train_dir}...")
    image_paths = get_image_paths(train_dir)
    
    if not image_paths:
        print("No images found in the training directory.")
        return
    
    print(f"Found {len(image_paths)} images.")
    
    # Calculate statistics (using a sample of 1000 images if dataset is large)
    sample_size = min(1000, len(image_paths))
    stats, channels = calculate_rgb_stats(image_paths, sample_size=sample_size)
    
    # Create and save statistics DataFrame
    stats_df = pd.DataFrame(stats).T
    stats_csv_path = os.path.join(output_dir, 'rgb_statistics.csv')
    stats_df.to_csv(stats_csv_path)
    print(f"Statistics saved to {stats_csv_path}")
    
    # Print statistics
    print("\nRGB Channel Statistics:")
    print("-" * 50)
    print(stats_df)
    
    # Plot and save distributions
    plot_path = os.path.join(output_dir, 'rgb_distributions.png')
    plot_channel_distributions(channels, save_path=plot_path)
    
    # Save statistics as text file
    with open(os.path.join(output_dir, 'rgb_statistics_summary.txt'), 'w') as f:
        f.write("RGB Channel Statistics\n")
        f.write("="*50 + "\n\n")
        f.write(stats_df.to_string())

if __name__ == "__main__":
    main()
