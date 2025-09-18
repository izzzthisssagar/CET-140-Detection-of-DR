import os
import numpy as np
import tensorflow as tf
import cv2

# Configuration
TEST_IMAGES_DIR = r"C:\Users\LOQ\Desktop\test_images"
OUTPUT_DIR = "gradcam_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Test script started")
print(f"Checking for images in: {TEST_IMAGES_DIR}")

# List files in test directory
try:
    files = os.listdir(TEST_IMAGES_DIR)
    print(f"Found {len(files)} files in test directory")
    print("First 5 files:", files[:5])
except Exception as e:
    print(f"Error listing directory: {e}")

print("Test script completed")
