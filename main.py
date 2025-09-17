# %% [markdown]
# # Diabetic Retinopathy Detection with Deep Learning
# 
# This project implements a deep learning system for detecting diabetic retinopathy from fundus images.
# The system uses transfer learning with advanced architectures and includes comprehensive evaluation.
# 
# ## Project Structure:
# 1. Environment setup and imports
# 2. Configuration and paths
# 3. Data preprocessing and organization
# 4. Data loading and preprocessing with augmentation
# 5. Visualization utilities
# 6. Model building functions
# 7. Training function with fine-tuning
# 8. Model evaluation
# 9. Main training execution
# 10. Grad-CAM implementation for interpretability
# 11. Comprehensive report generation
# 12. Final deployment model
# 
# Author: SAGAR THAPA (bb955)
# Module: CET140 Specialist Project
# Date: 2025

# %%
# CELL 1: Environment setup and imports
# ---------------------------------------------------------------------------
# Import necessary libraries for data processing, visualization, and deep learning
import os
# Force CPU usage to bypass GPU initialization errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import shutil
from pathlib import Path
import cv2
from PIL import Image
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

# Print TensorFlow version for reproducibility
print("TensorFlow version:", tf.__version__)

# Minimal device diagnostics (safe)
try:
    devices = tf.config.list_physical_devices()
    print("Detected devices:")
    for d in devices:
        print(" -", d)
except Exception as _e:
    # Avoid crashing on diagnostics
    pass

# Check for GPU availability and configure for optimal performance
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f'GPU(s) found: {len(gpus)}')
        # Set mixed precision policy for better performance on supported GPUs
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled for GPU training")
    except Exception as e:
        print('Could not set memory growth:', e)
else:
    print('No GPUs found - using CPU for training')

# %%
# CELL 2: Configuration and paths
# ---------------------------------------------------------------------------
# Set base directory and paths for data organization
BASE_DIR = r"c:\Users\LOQ\Desktop\DR_Project\data"  # Change this to your project directory
RAW_IMAGES_DIR = os.path.join(BASE_DIR, "raw_images")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_images")
TRAIN_DIR = os.path.join(BASE_DIR, "train_images")
TEST_DIR = os.path.join(BASE_DIR, "test_images")
RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")

# Create directories if they don't exist
os.makedirs(RAW_IMAGES_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Print directory information for verification
print(f"Base directory: {BASE_DIR}")
print(f"Raw images: {RAW_IMAGES_DIR}")
print(f"Processed images: {PROCESSED_DIR}")
print(f"Train directory: {TRAIN_DIR}")
print(f"Test directory: {TEST_DIR}")
print(f"Results directory: {RESULTS_DIR}")

# Image parameters - optimized for transfer learning models
IMG_SIZE = (299, 299)  # Standard size for Inception and similar architectures
SEED = 42  # Random seed for reproducibility
BATCH_SIZE = 32  # Optimal batch size for GPU memory
AUTOTUNE = tf.data.AUTOTUNE  # Let TensorFlow optimize data loading
EPOCHS = 25  # Total training epochs (will be split between frozen and fine-tuning)

# Quick functional test configuration
# Set QUICK_RUN=True to train a single model with fewer epochs to validate the pipeline quickly
QUICK_RUN = False
QUICK_MODEL = 'ResNet50'  # Options: 'ResNet50', 'EfficientNetB4', 'InceptionV3', 'DenseNet201'
QUICK_EPOCHS = 6               # total epochs for quick run
QUICK_FINE_TUNE_EPOCHS = 2     # fine-tuning epochs for quick run

# %%
# CELL 3: Data preprocessing and organization
# ---------------------------------------------------------------------------
def preprocess_aptos_image(image_path, target_size=IMG_SIZE):
    """
    Preprocess APTOS fundus images with advanced enhancements
    
    Parameters:
    - image_path: Path to the input image
    - target_size: Target size for resizing (default: IMG_SIZE)
    
    Returns:
    - Preprocessed image as numpy array or None if error occurs
    """
    try:
        # Read image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            return None
            
        # Convert from BGR to RGB format (OpenCV uses BGR, models expect RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement
        # This helps improve visibility of blood vessels and lesions
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # Resize image to target size using interpolation for quality preservation
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
        
        # Normalize pixel values to [0, 1] range for neural network compatibility
        img = img.astype(np.float32) / 255.0
        
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def organize_aptos_dataset(raw_dir, processed_dir, train_dir, test_dir, csv_filename="train.csv", train_split=0.8):
    """
    Organize APTOS 2019 dataset into binary classification (No DR vs DR)
    
    Parameters:
    - raw_dir: Directory containing raw images
    - processed_dir: Directory to save processed images
    - train_dir: Directory for training images
    - test_dir: Directory for test images
    - csv_filename: Name of the CSV file with labels
    - train_split: Proportion of data to use for training (default: 0.8)
    
    Returns:
    - Boolean indicating success or failure
    """
    # Check if CSV file exists
    csv_path = os.path.join(raw_dir, csv_filename)
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} not found!")
        print("Please ensure train.csv is in the raw_images directory")
        return False
    
    # Load CSV data into pandas DataFrame
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from CSV")
    
    # Convert to binary classification: 0 = No DR, 1 = DR (levels 1-4)
    # This simplifies the problem while maintaining clinical relevance
    df['binary_label'] = df['diagnosis'].apply(lambda x: 0 if x == 0 else 1)
    
    # Check class distribution to understand data imbalance
    print("Class distribution:")
    print(df['binary_label'].value_counts())
    print("\nOriginal severity distribution:")
    print(df['diagnosis'].value_counts().sort_index())
    
    # Create directories for organized data
    for label in [0, 1]:
        os.makedirs(os.path.join(train_dir, str(label)), exist_ok=True)
        os.makedirs(os.path.join(test_dir, str(label)), exist_ok=True)
    
    # Process and organize images with progress tracking
    np.random.seed(SEED)
    image_files = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        img_name = f"{row['id_code']}.png"
        img_path = os.path.join(raw_dir, "train_images", img_name)
        
        # Skip if image file doesn't exist
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
            
        # Preprocess image using our enhancement pipeline
        processed_img = preprocess_aptos_image(img_path)
        if processed_img is None:
            continue
            
        # Save processed image for future use
        processed_img_path = os.path.join(processed_dir, img_name)
        cv2.imwrite(processed_img_path, (processed_img * 255).astype(np.uint8))
        
        # Store image information for dataset organization
        image_files.append({
            'path': processed_img_path,
            'label': row['binary_label'],
            'original_label': row['diagnosis']
        })
    
    # Shuffle and split data into training and test sets
    np.random.shuffle(image_files)
    split_idx = int(len(image_files) * train_split)
    train_images = image_files[:split_idx]
    test_images = image_files[split_idx:]
    
    # Copy images to appropriate directories
    for img_info in tqdm(train_images, desc="Organizing training images"):
        dst = os.path.join(train_dir, str(img_info['label']), os.path.basename(img_info['path']))
        shutil.copy2(img_info['path'], dst)
    
    for img_info in tqdm(test_images, desc="Organizing test images"):
        dst = os.path.join(test_dir, str(img_info['label']), os.path.basename(img_info['path']))
        shutil.copy2(img_info['path'], dst)
    
    print(f"Dataset organized: {len(train_images)} training, {len(test_images)} test images")
    
    # Save dataset information for documentation and reproducibility
    dataset_info = {
        'train_count': len(train_images),
        'test_count': len(test_images),
        'class_distribution': {
            'train_0': len([img for img in train_images if img['label'] == 0]),
            'train_1': len([img for img in train_images if img['label'] == 1]),
            'test_0': len([img for img in test_images if img['label'] == 0]),
            'test_1': len([img for img in test_images if img['label'] == 1]),
        }
    }
    
    # Write dataset info to file
    with open(os.path.join(LOGS_DIR, "dataset_info.txt"), "w") as f:
        for key, value in dataset_info.items():
            f.write(f"{key}: {value}\n")
    
    return True

# Uncomment and run this function after placing your dataset in the raw_images folder
organize_aptos_dataset(RAW_IMAGES_DIR, PROCESSED_DIR, TRAIN_DIR, TEST_DIR)

# %%
# CELL 4: Data loading and preprocessing with augmentation
# ---------------------------------------------------------------------------
def create_datasets():
    """
    Create TensorFlow datasets from organized images with augmentation
    
    Returns:
    - train_ds: Training dataset
    - val_ds: Validation dataset
    - test_ds: Test dataset
    - class_weights: Class weights for handling imbalanced data
    """
    # Check if we have images in the train directory
    train_0_count = len(os.listdir(os.path.join(TRAIN_DIR, "0"))) if os.path.exists(os.path.join(TRAIN_DIR, "0")) else 0
    train_1_count = len(os.listdir(os.path.join(TRAIN_DIR, "1"))) if os.path.exists(os.path.join(TRAIN_DIR, "1")) else 0
    
    # If no images found, provide instructions
    if train_0_count == 0 and train_1_count == 0:
        print("""
        ERROR: No training images found. Please:
        1. Place your APTOS 2019 dataset in the raw_images folder
        2. Ensure it has train.csv and train_images folder
        3. Run the organize_aptos_dataset() function in the previous cell
        """)
        return None, None, None, None
    
    print(f"Training images: {train_0_count} non-DR, {train_1_count} DR")
    
    # Calculate class weights for imbalanced data
    # This helps the model pay more attention to underrepresented classes
    class_weights = compute_class_weight(
        'balanced',
        classes=np.array([0, 1]),
        y=np.concatenate([np.zeros(train_0_count), np.ones(train_1_count)])
    )
    class_weights = dict(enumerate(class_weights))
    print(f"Class weights: {class_weights}")
    
    # Data augmentation for training - improves generalization
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
    ], name="data_augmentation")
    
    # Preprocessing function with optional augmentation
    def preprocess(image, label, augment=True):
        if augment:
            image = data_augmentation(image, training=True)
        return image, label
    
    # Create training dataset with 80/20 train/validation split
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        labels='inferred',
        label_mode='binary',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=0.2,
        subset='training'
    )

    # Create validation dataset
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        labels='inferred',
        label_mode='binary',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=0.2,
        subset='validation'
    )

    # Create test dataset
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_DIR,
        labels='inferred',
        label_mode='binary',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Apply preprocessing with augmentation to training data only
    train_ds = train_ds.map(lambda x, y: preprocess(x, y, augment=True))
    val_ds = val_ds.map(lambda x, y: preprocess(x, y, augment=False))
    test_ds = test_ds.map(lambda x, y: preprocess(x, y, augment=False))
    
    # Prefetch for performance optimization
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    # Print dataset information
    print(f'Train batches: {tf.data.experimental.cardinality(train_ds).numpy()}')
    print(f'Val batches: {tf.data.experimental.cardinality(val_ds).numpy()}')
    print(f'Test batches: {tf.data.experimental.cardinality(test_ds).numpy()}')
    
    return train_ds, val_ds, test_ds, class_weights

# Create datasets
train_ds, val_ds, test_ds, class_weights = create_datasets()

# %%
# CELL 5: Visualization utilities
# ---------------------------------------------------------------------------
def plot_sample_images(dataset, class_names=['Non-DR', 'DR']):
    """
    Plot sample images from dataset for visual inspection
    
    Parameters:
    - dataset: TensorFlow dataset to sample from
    - class_names: List of class names for labeling
    """
    plt.figure(figsize=(12, 10))
    for images, labels in dataset.take(1):
        for i in range(12):
            ax = plt.subplot(3, 4, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(f"{class_names[int(labels[i])]}")
            plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "sample_images.png"), dpi=300, bbox_inches='tight')
    plt.show()

def plot_class_distribution_pie():
    """Plot class distribution as pie charts for training and test sets"""
    train_0 = len(os.listdir(os.path.join(TRAIN_DIR, "0"))) if os.path.exists(os.path.join(TRAIN_DIR, "0")) else 0
    train_1 = len(os.listdir(os.path.join(TRAIN_DIR, "1"))) if os.path.exists(os.path.join(TRAIN_DIR, "1")) else 0
    test_0 = len(os.listdir(os.path.join(TEST_DIR, "0"))) if os.path.exists(os.path.join(TEST_DIR, "0")) else 0
    test_1 = len(os.listdir(os.path.join(TEST_DIR, "1"))) if os.path.exists(os.path.join(TEST_DIR, "1")) else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Training data distribution pie chart
    ax1.pie([train_0, train_1], labels=['Non-DR', 'DR'], autopct='%1.1f%%', colors=['skyblue', 'salmon'], startangle=90)
    ax1.set_title('Training Data Distribution')

    # Test data distribution pie chart
    ax2.pie([test_0, test_1], labels=['Non-DR', 'DR'], autopct='%1.1f%%', colors=['skyblue', 'salmon'], startangle=90)
    ax2.set_title('Test Data Distribution')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "class_distribution_pie.png"), dpi=300, bbox_inches='tight')
    plt.show()

def plot_class_distribution():
    """Plot class distribution for training and test sets"""
    train_0 = len(os.listdir(os.path.join(TRAIN_DIR, "0"))) if os.path.exists(os.path.join(TRAIN_DIR, "0")) else 0
    train_1 = len(os.listdir(os.path.join(TRAIN_DIR, "1"))) if os.path.exists(os.path.join(TRAIN_DIR, "1")) else 0
    test_0 = len(os.listdir(os.path.join(TEST_DIR, "0"))) if os.path.exists(os.path.join(TEST_DIR, "0")) else 0
    test_1 = len(os.listdir(os.path.join(TEST_DIR, "1"))) if os.path.exists(os.path.join(TEST_DIR, "1")) else 0
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training data distribution
    ax1.bar(['Non-DR', 'DR'], [train_0, train_1], color=['green', 'red'])
    ax1.set_title('Training Data Distribution')
    ax1.set_ylabel('Count')
    
    # Test data distribution
    ax2.bar(['Non-DR', 'DR'], [test_0, test_1], color=['green', 'red'])
    ax2.set_title('Test Data Distribution')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "class_distribution_bar.png"), dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_history(history, model_name):
    """
    Plot training history with multiple metrics
    
    Parameters:
    - history: Training history object
    - model_name: Name of the model for title
    """
    # Support both keras.callbacks.History objects and plain dicts
    hist_dict = history.history if hasattr(history, 'history') else history
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(hist_dict.get('loss', []), label='Training Loss')
    axes[0, 0].plot(hist_dict.get('val_loss', []), label='Validation Loss')
    axes[0, 0].set_title(f'{model_name} - Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy plot
    axes[0, 1].plot(hist_dict.get('accuracy', []), label='Training Accuracy')
    axes[0, 1].plot(hist_dict.get('val_accuracy', []), label='Validation Accuracy')
    axes[0, 1].set_title(f'{model_name} - Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision plot (if available)
    if 'precision' in hist_dict:
        axes[1, 0].plot(hist_dict.get('precision', []), label='Training Precision')
        axes[1, 0].plot(hist_dict.get('val_precision', []), label='Validation Precision')
        axes[1, 0].set_title(f'{model_name} - Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Recall plot (if available)
    if 'recall' in hist_dict:
        axes[1, 1].plot(hist_dict.get('recall', []), label='Training Recall')
        axes[1, 1].plot(hist_dict.get('val_recall', []), label='Validation Recall')
        axes[1, 1].set_title(f'{model_name} - Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'{model_name}_training_history.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Plot confusion matrix for model evaluation
    
    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - model_name: Name of the model for title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-DR', 'DR'], 
                yticklabels=['Non-DR', 'DR'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'{model_name}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curve(y_true, y_pred_probs, model_name):
    """
    Plot ROC curve for model evaluation
    
    Parameters:
    - y_true: True labels
    - y_pred_probs: Predicted probabilities
    - model_name: Name of the model for title
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'{model_name}_roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.show()

# Plot sample images and class distribution if datasets are available
if train_ds:
    plot_sample_images(train_ds)
    plot_class_distribution()
    plot_class_distribution_pie()

# %%
# CELL 6: Model building functions with advanced architectures
# ---------------------------------------------------------------------------
def create_advanced_model(base_model_name, input_shape=(299, 299, 3)):
    """
    Create an advanced transfer learning model with fine-tuning capability
    
    Parameters:
    - base_model_name: Name of the base model architecture
    - input_shape: Input shape for the model
    
    Returns:
    - model: Compiled Keras model
    - base_model: Base model for fine-tuning
    """
    # Select base model based on name
    if base_model_name == 'ResNet50':
        base_model = tf.keras.applications.ResNet50V2(
            weights='imagenet', 
            include_top=False, 
            input_shape=input_shape
        )
    elif base_model_name == 'EfficientNetB0':
        # Use random init to avoid occasional pretrained weight shape mismatches on some envs
        base_model = tf.keras.applications.EfficientNetB0(
            weights=None,
            include_top=False,
            input_shape=input_shape
        )
    elif base_model_name == 'InceptionV3':
        base_model = tf.keras.applications.InceptionV3(
            weights='imagenet', 
            include_top=False, 
            input_shape=input_shape
        )
    elif base_model_name == 'DenseNet201':
        base_model = tf.keras.applications.DenseNet201(
            weights='imagenet', 
            include_top=False, 
            input_shape=input_shape
        )
    else:
        raise ValueError(f"Unknown model: {base_model_name}")
    
    # Freeze base model initially (transfer learning approach)
    base_model.trainable = False
    
    # Build custom top layers for our specific task
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)  # Regularization to prevent overfitting
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)  # Stabilize training
    x = tf.keras.layers.Dropout(0.3)(x)  # Additional regularization
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)  # Binary classification
    
    model = tf.keras.Model(inputs, outputs)
    
    return model, base_model

def compile_model(model, learning_rate=1e-4):
    """
    Compile model with appropriate settings
    
    Parameters:
    - model: Keras model to compile
    - learning_rate: Learning rate for optimizer
    
    Returns:
    - Compiled model
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',  # Standard for binary classification
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(name='precision'),  # Important for medical applications
                 tf.keras.metrics.Recall(name='recall'),       # Important for medical applications
                 tf.keras.metrics.AUC(name='auc')]             # Comprehensive performance metric
    )
    return model

def unfreeze_layers(model, base_model, unfreeze_last_n=20):
    """
    Unfreeze the last n layers of the base model for fine-tuning
    
    Parameters:
    - model: Full model
    - base_model: Base model part
    - unfreeze_last_n: Number of layers to unfreeze from the end
    
    Returns:
    - Recompiled model with unfrozen layers
    """
    # Make base model trainable
    base_model.trainable = True
    
    # Freeze all layers except the last n for fine-tuning
    for layer in base_model.layers[:-unfreeze_last_n]:
        layer.trainable = False
        
    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    
    print(f"Unfroze last {unfreeze_last_n} layers for fine-tuning")
    return model

# %%
# CELL 7: Training function with fine-tuning
# ---------------------------------------------------------------------------
def train_model_with_finetuning(model_name, train_data, val_data, class_weights, epochs=EPOCHS, fine_tune_epochs=10):
    """
    Train a model with fine-tuning and return history
    
    Parameters:
    - model_name: Name of the model architecture
    - train_data: Training dataset
    - val_data: Validation dataset
    - class_weights: Class weights for imbalanced data
    - epochs: Total training epochs
    - fine_tune_epochs: Number of epochs for fine-tuning phase
    
    Returns:
    - model: Trained model
    - combined_history: Combined training history
    """
    print(f"\nTraining {model_name}...")
    
    # Create and compile model
    model, base_model = create_advanced_model(model_name, input_shape=(*IMG_SIZE, 3))
    model = compile_model(model)
    
    # Callbacks for training optimization
    callbacks = [
        # Stop training if no improvement after 5 epochs
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_auc', mode='max'),
        # Reduce learning rate when validation metric plateaus
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7),
        # Save best model based on validation AUC
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, f"{model_name}_best.keras"),
            save_best_only=True,
            monitor='val_auc',
            mode='max',
            verbose=1
        ),
        # Log training history to CSV
        tf.keras.callbacks.CSVLogger(os.path.join(LOGS_DIR, f"{model_name}_history.csv"))
    ]
    
    # First phase: train only the top layers (transfer learning)
    print("Phase 1: Training top layers...")
    history1 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs - fine_tune_epochs,
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weights  # Handle class imbalance
    )
    
    # Second phase: fine-tune the last layers of the base model
    print("Phase 2: Fine-tuning last layers...")
    model = unfreeze_layers(model, base_model)
    
    history2 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=fine_tune_epochs,
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weights,
        initial_epoch=history1.epoch[-1] + 1 if history1.epoch else 0
    )
    
    # Combine histories from both phases safely (handle missing keys)
    combined_history = {}
    all_keys = set(history1.history.keys()) | set(history2.history.keys())
    for key in all_keys:
        h1 = history1.history.get(key, [])
        h2 = history2.history.get(key, [])
        # Ensure list types for concatenation
        if not isinstance(h1, list):
            h1 = list(h1)
        if not isinstance(h2, list):
            h2 = list(h2)
        combined_history[key] = h1 + h2
    # Persist combined history to CSV as a fallback to CSVLogger
    try:
        import csv
        epochs_len = max((len(v) for v in combined_history.values()), default=0)
        keys = sorted(combined_history.keys())
        out_csv = os.path.join(LOGS_DIR, f"{model_name}_history.csv")
        with open(out_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch'] + keys)
            for i in range(epochs_len):
                row = [i+1]
                for k in keys:
                    vals = combined_history.get(k, [])
                    row.append(vals[i] if i < len(vals) else '')
                writer.writerow(row)
    except Exception as e:
        print(f"Warning: could not write combined history CSV for {model_name}: {e}")
    
    return model, combined_history

# %%
# CELL 8: Model evaluation with comprehensive metrics
# ---------------------------------------------------------------------------
def evaluate_model(model, test_data, model_name):
    """
    Evaluate model and generate comprehensive reports
    
    Parameters:
    - model: Trained model to evaluate
    - test_data: Test dataset
    - model_name: Name of the model for reporting
    
    Returns:
    - Dictionary with evaluation results
    """
    print(f"\nEvaluating {model_name}...")
    
    # Evaluate on test set
    results = model.evaluate(test_data, verbose=0)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    print(f"Test Precision: {results[2]:.4f}")
    print(f"Test Recall: {results[3]:.4f}")
    print(f"Test AUC: {results[4]:.4f}")
    
    # Generate predictions for detailed analysis
    y_true = []
    y_pred_probs = []
    
    for images, labels in test_data:
        y_true.extend(labels.numpy())
        predictions = model.predict(images, verbose=0)
        y_pred_probs.extend(predictions.flatten())
    
    # Convert probabilities to binary predictions
    y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_probs]
    
    # Classification report with precision, recall, f1-score
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_true, y_pred, target_names=['Non-DR', 'DR']))
    
    # Visualization of results
    plot_confusion_matrix(y_true, y_pred, model_name)
    plot_roc_curve(y_true, y_pred_probs, model_name)
    
    # Save results for later analysis
    results_dict = {
        'loss': results[0],
        'accuracy': results[1],
        'precision': results[2],
        'recall': results[3],
        'auc': results[4],
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_probs': y_pred_probs
    }
    
    # Save predictions to CSV file
    results_df = pd.DataFrame({
        'true_label': y_true,
        'predicted_label': y_pred,
        'prediction_probability': y_pred_probs
    })
    results_df.to_csv(os.path.join(LOGS_DIR, f"{model_name}_predictions.csv"), index=False)
    
    return results_dict

# %%
# CELL 9: Main training execution
# ---------------------------------------------------------------------------
def main():
    """
    Main function to run the complete training pipeline
    
    Returns:
    - trained_models: Dictionary of trained models
    - histories: Dictionary of training histories
    - results: Dictionary of evaluation results
    - best_model_name: Name of the best performing model
    """
    # Check if we have data
    if train_ds is None:
        print("No data available. Please organize your dataset first.")
        return None, None, None, None
    
    # Models to train (using proven architectures for medical imaging)
    if QUICK_RUN:
        print("QUICK_RUN is enabled: training a single model with fewer epochs for a functional test.")
        models_to_train = [QUICK_MODEL]
        epochs_to_use = QUICK_EPOCHS
        fine_tune_to_use = QUICK_FINE_TUNE_EPOCHS
    else:
        models_to_train = ['ResNet50', 'InceptionV3', 'EfficientNetB0']
        epochs_to_use = EPOCHS
        fine_tune_to_use = 10
    trained_models = {}
    histories = {}
    results = {}
    
    # Train each model
    for model_name in models_to_train:
        try:
            model, history = train_model_with_finetuning(
                model_name,
                train_ds,
                val_ds,
                class_weights,
                epochs=epochs_to_use,
                fine_tune_epochs=fine_tune_to_use
            )
            trained_models[model_name] = model
            histories[model_name] = history

            # Plot training history for analysis
            plot_training_history(history, model_name)

            # Evaluate model on test set
            results[model_name] = evaluate_model(model, test_ds, model_name)

            # Save model for future use (robust to TF/Keras save format differences)
            model_path = os.path.join(MODELS_DIR, f"{model_name}_model")
            try:
                # Try SavedModel directory format
                model.save(model_path)
                print(f"Saved {model_name} model to {model_path}")
            except Exception as e:
                # Fallback to native Keras format
                keras_path = model_path + ".keras"
                try:
                    model.save(keras_path)
                    print(f"Saved {model_name} model to {keras_path} (fallback)")
                except Exception as e2:
                    print(f"ERROR: Failed to save {model_name} model. Primary error: {e}. Fallback error: {e2}")
        except Exception as e:
            print(f"ERROR while training {model_name}: {e}. Skipping to next model.")
    
    # Compare models to select the best one
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    
    comparison_data = []
    for model_name, res in results.items():
        acc = res['accuracy']
        prec = res['precision']
        rec = res['recall']
        auc_score = res['auc']
        print(f"{model_name}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, AUC={auc_score:.4f}")
        comparison_data.append([model_name, acc, prec, rec, auc_score])
    
    # Save comparison results to CSV
    results_file = os.path.join(LOGS_DIR, "model_comparison.csv")
    df = pd.DataFrame(comparison_data, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'AUC'])
    df.to_csv(results_file, index=False)
    
    print(f"\nResults saved to {results_file}")
    
    # Visualize model comparison
    plt.figure(figsize=(12, 8))
    x = np.arange(len(comparison_data))
    width = 0.2
    
    plt.bar(x - width*1.5, df['Accuracy'], width, label='Accuracy', alpha=0.8)
    plt.bar(x - width/2, df['Precision'], width, label='Precision', alpha=0.8)
    plt.bar(x + width/2, df['Recall'], width, label='Recall', alpha=0.8)
    plt.bar(x + width*1.5, df['AUC'], width, label='AUC', alpha=0.8)
    
    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.title('Model Performance Comparison')
    plt.xticks(x, [row[0] for row in comparison_data])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # Select best model based on AUC (most important metric for medical applications)
    best_model_name = max(results.items(), key=lambda x: x[1]['auc'])[0]
    print(f"\nBest model: {best_model_name} with AUC: {results[best_model_name]['auc']:.4f}")
    
    return trained_models, histories, results, best_model_name

# %%
# CELL 10: Grad-CAM implementation for model interpretability
# ---------------------------------------------------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generate Grad-CAM heatmap for model interpretability
    
    Parameters:
    - img_array: Input image array
    - model: Trained model
    - last_conv_layer_name: Name of the last convolutional layer
    - pred_index: Index of predicted class (None for auto-detection)
    
    Returns:
    - Heatmap showing important regions for prediction
    """
    # Create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradient of the output neuron with regard to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Vector where each entry is the mean intensity of the gradient
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by "how important this channel is"
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap for visualization
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, alpha=0.4):
    """
    Save and display Grad-CAM visualization
    
    Parameters:
    - img_path: Path to the original image
    - heatmap: Grad-CAM heatmap
    - alpha: Transparency for heatmap overlay
    
    Returns:
    - Path to the saved visualization
    """
    # Load the original image
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = plt.colormaps.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    cam_path = os.path.join(PLOTS_DIR, "grad_cam_visualization.png")
    superimposed_img.save(cam_path)
    
    return cam_path

def _find_last_conv2d_layer_name(model):
    """
    Recursively search for the last Conv2D layer name within a (possibly nested) model.
    Returns the layer name or None if not found.
    """
    last_name = None
    # Depth-first search through layers and sublayers
    def visit(layer):
        nonlocal last_name
        try:
            import tensorflow as tf  # local import to avoid issues if function is reused
            from tensorflow.keras.layers import Conv2D
        except Exception:
            return
        if isinstance(layer, Conv2D):
            last_name = layer.name
        # If the layer itself is a Model (has sublayers), visit them
        if hasattr(layer, 'layers'):
            for sub in layer.layers:
                visit(sub)
    visit(model)
    return last_name

def visualize_model_interpretability(model, test_data, model_name, num_images=5):
    """
    Create Grad-CAM visualizations for model interpretability
    
    Parameters:
    - model: Trained model
    - test_data: Test dataset
    - model_name: Name of the model
    - num_images: Number of images to visualize
    """
    # Resolve the last convolutional layer name robustly
    last_conv_layer_name = _find_last_conv2d_layer_name(model)
    if not last_conv_layer_name:
        print("Warning: No Conv2D layer found for Grad-CAM. Skipping interpretability visualization.")
        return
    print(f"Using layer {last_conv_layer_name} for Grad-CAM")
    
    # Prepare output directory for Grad-CAMs
    gradcam_dir = os.path.join(PLOTS_DIR, "gradcam")
    os.makedirs(gradcam_dir, exist_ok=True)

    if num_images is None:
        # Process the entire test dataset
        idx = 0
        for images, labels in test_data:
            batch_size = images.shape[0]
            for b in range(batch_size):
                img = images[b]
                img_array = tf.expand_dims(img, axis=0)
                heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
                temp_img_path = os.path.join(gradcam_dir, f"temp_img_{idx}.png")
                tf.keras.preprocessing.image.save_img(temp_img_path, img)
                cam_path = save_and_display_gradcam(temp_img_path, heatmap)
                # Move/rename into gradcam subfolder per model
                final_cam = os.path.join(gradcam_dir, f"{model_name}_gradcam_{idx}.png")
                try:
                    os.replace(cam_path, final_cam)
                except Exception:
                    # If replace fails across FS boundaries
                    import shutil as _shutil
                    _shutil.copyfile(cam_path, final_cam)
                    os.remove(cam_path)
                os.remove(temp_img_path)
                idx += 1
        print(f"Saved {idx} Grad-CAM images to {gradcam_dir}")
    else:
        # Get sample images from test set
        images, labels = next(iter(test_data.unbatch().batch(num_images)))
        
        # Create visualization for each image
        for i in range(min(num_images, len(images))):
            img_array = tf.expand_dims(images[i], axis=0)
            
            # Generate heatmap
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
            
            # Create temporary image file
            temp_img_path = os.path.join(gradcam_dir, f"temp_img_{i}.png")
            tf.keras.preprocessing.image.save_img(temp_img_path, images[i])
            
            # Save Grad-CAM visualization
            cam_path = save_and_display_gradcam(temp_img_path, heatmap)
            final_cam = os.path.join(gradcam_dir, f"{model_name}_gradcam_{i}.png")
            try:
                os.replace(cam_path, final_cam)
            except Exception:
                import shutil as _shutil
                _shutil.copyfile(cam_path, final_cam)
                os.remove(cam_path)
            print(f"Grad-CAM visualization saved to {final_cam}")
            
            # Clean up temporary file
            os.remove(temp_img_path)

# %%
# CELL 11: Create a comprehensive report
# ---------------------------------------------------------------------------
def create_comprehensive_report(results, best_model_name):
    """
    Create a comprehensive PDF report of the project
    
    Parameters:
    - results: Dictionary with evaluation results
    - best_model_name: Name of the best performing model
    """
    from fpdf import FPDF
    
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'Diabetic Retinopathy Detection Project Report', 0, 1, 'C')
            self.ln(5)
        
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    # Create PDF
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Diabetic Retinopathy Detection Using Deep Learning', 0, 1, 'C')
    pdf.ln(10)
    
    # Add project details
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Student: SAGAR THAPA (bb955)', 0, 1)
    pdf.cell(0, 10, f'Module: CET140 Specialist Project', 0, 1)
    pdf.cell(0, 10, f'Date: {pd.Timestamp.now().strftime("%Y-%m-%d")}', 0, 1)
    pdf.ln(10)
    
    # Add dataset information
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Dataset Information', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    train_0 = len(os.listdir(os.path.join(TRAIN_DIR, "0"))) if os.path.exists(os.path.join(TRAIN_DIR, "0")) else 0
    train_1 = len(os.listdir(os.path.join(TRAIN_DIR, "1"))) if os.path.exists(os.path.join(TRAIN_DIR, "1")) else 0
    test_0 = len(os.listdir(os.path.join(TEST_DIR, "0"))) if os.path.exists(os.path.join(TEST_DIR, "0")) else 0
    test_1 = len(os.listdir(os.path.join(TEST_DIR, "1"))) if os.path.exists(os.path.join(TEST_DIR, "1")) else 0
    
    pdf.cell(0, 10, f'Training images: {train_0} Non-DR, {train_1} DR', 0, 1)
    pdf.cell(0, 10, f'Test images: {test_0} Non-DR, {test_1} DR', 0, 1)
    pdf.ln(10)
    
    # Add model comparison
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Model Performance Comparison', 0, 1)
    
    # Create a table
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(40, 10, 'Model', 1, 0, 'C')
    pdf.cell(30, 10, 'Accuracy', 1, 0, 'C')
    pdf.cell(30, 10, 'Precision', 1, 0, 'C')
    pdf.cell(30, 10, 'Recall', 1, 0, 'C')
    pdf.cell(30, 10, 'AUC', 1, 1, 'C')
    
    pdf.set_font('Arial', '', 12)
    for model_name in results.keys():
        pdf.cell(40, 10, model_name, 1, 0, 'C')
        pdf.cell(30, 10, f"{results[model_name]['accuracy']:.4f}", 1, 0, 'C')
        pdf.cell(30, 10, f"{results[model_name]['precision']:.4f}", 1, 0, 'C')
        pdf.cell(30, 10, f"{results[model_name]['recall']:.4f}", 1, 0, 'C')
        pdf.cell(30, 10, f"{results[model_name]['auc']:.4f}", 1, 1, 'C')
    
    pdf.ln(10)
    pdf.cell(0, 10, f'Best Model: {best_model_name} (AUC: {results[best_model_name]["auc"]:.4f})', 0, 1)
    pdf.ln(10)
    
    # Add images to the report
    image_files = [
        "class_distribution_bar.png",
        "class_distribution_pie.png",
        "sample_images.png",
        f"{best_model_name}_training_history.png",
        f"{best_model_name}_confusion_matrix.png",
        f"{best_model_name}_roc_curve.png",
        "model_comparison.png"
    ]
    
    for img_file in image_files:
        img_path = os.path.join(PLOTS_DIR, img_file)
        if os.path.exists(img_path):
            pdf.add_page()
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, img_file.replace('.png', '').replace('_', ' ').title(), 0, 1)
            pdf.image(img_path, x=10, y=30, w=180)
            pdf.ln(120)
    
    # Save PDF
    pdf_path = os.path.join(REPORTS_DIR, "project_report.pdf")
    pdf.output(pdf_path)
    print(f"Comprehensive report saved to {pdf_path}")

# %%
# CELL 12: Final deployment model
# ---------------------------------------------------------------------------
def create_deployment_model(best_model_name, trained_models):
    """
    Create a final deployment model with the best architecture
    
    Parameters:
    - best_model_name: Name of the best model
    - trained_models: Dictionary of trained models
    
    Returns:
    - Deployment-ready model
    """
    best_model = trained_models[best_model_name]
    
    # Save the final model in multiple formats
    model_path = os.path.join(MODELS_DIR, "deployment_model")
    best_model.save(model_path)
    print(f"Deployment model saved to {model_path}")
    
    # Convert to TFLite for mobile deployment (optional)
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        tflite_path = os.path.join(MODELS_DIR, "deployment_model.tflite")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"TFLite model saved to {tflite_path}")
    except Exception as e:
        print(f"Could not convert to TFLite: {e}")
    
    return best_model

def predict_on_external_images(model, external_images_dir, output_csv_path):
    """
    Run predictions on a directory of external images and save the results.

    Parameters:
    - model: The trained model to use for predictions.
    - external_images_dir: Path to the directory containing external test images.
    - output_csv_path: Path to save the output CSV file.
    """
    print(f"\nRunning predictions on external images in: {external_images_dir}")
    image_files = [f for f in os.listdir(external_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    predictions = []

    for img_name in tqdm(image_files, desc="Predicting on external images"):
        img_path = os.path.join(external_images_dir, img_name)
        processed_img = preprocess_aptos_image(img_path)
        if processed_img is not None:
            img_array = np.expand_dims(processed_img, axis=0)
            prediction_prob = model.predict(img_array, verbose=0)[0][0]
            prediction_label = 1 if prediction_prob > 0.5 else 0
            predictions.append({
                'image_id': img_name,
                'prediction_probability': prediction_prob,
                'predicted_label': prediction_label,
                'diagnosis': 'Diabetic Retinopathy' if prediction_label == 1 else 'No Diabetic Retinopathy'
            })

    # Save predictions to CSV
    df = pd.DataFrame(predictions)
    df.to_csv(output_csv_path, index=False)
    print(f"External predictions saved to: {output_csv_path}")

if __name__ == "__main__":
    # Run the main training pipeline
    trained_models, histories, results, best_model_name = main()

    # After training, generate a comprehensive report and visualizations
    if results and trained_models and best_model_name:
        print("\n" + "="*60)
        print("POST-TRAINING ANALYSIS")
        print("="*60)
        
        # Get the best performing model
        best_model = trained_models[best_model_name]
        
        # Visualize model interpretability for the best model
        print("\nVisualizing model interpretability...")
        # Generate Grad-CAM for all test images
        visualize_model_interpretability(best_model, test_ds, best_model_name, num_images=None)
        
        # Create a comprehensive report
        print("\nCreating comprehensive report...")
        create_comprehensive_report(results, best_model_name)
        
        # Create and save the final deployment model
        print("\nCreating deployment model...")
        deployment_model = create_deployment_model(best_model_name, trained_models)

        # Run predictions on external test images
        external_images_dir = r"c:\Users\LOQ\Desktop\test_images"
        external_predictions_csv = os.path.join(REPORTS_DIR, "external_test_images_predictions.csv")
        predict_on_external_images(deployment_model, external_images_dir, external_predictions_csv)

        # Final message
        print("\n" + "="*60)
        print("PROJECT EXECUTION COMPLETE")
        print("="*60)
        print(f"All results have been saved to: {RESULTS_DIR}")
        print("Check the subdirectories for:")
        print("- models: Trained models")
        print("- plots: All generated graphs and charts")
        print("- reports: PDF summary and prediction CSVs")
        print("- logs: Training history and other logs")
        print("\nNext steps:")
        print("1. Review the project_report.pdf in the 'reports' folder.")
        print("2. Use the best model for future predictions.")
        print("3. Prepare your presentation using the generated plots and results.")
    else:
        print("Training failed or was skipped. Post-processing steps will not be executed.")

# Utility: finalize post-processing using existing artifacts (no re-training)
def run_post_training_from_artifacts(external_images_dir=r"c:\\Users\\LOQ\\Desktop\\test_images"):
    """
    Use saved artifacts to run post-processing steps without re-training:
    - Read best model from model_comparison.csv
    - Load best model from results/models
    - Rebuild test dataset quickly
    - Generate Grad-CAM for all test images
    - Run external predictions
    - Generate comprehensive report
    """
    # 1) Read comparison to determine best model
    comp_csv = os.path.join(LOGS_DIR, "model_comparison.csv")
    if not os.path.exists(comp_csv):
        print(f"Cannot run post-processing: {comp_csv} not found.")
        return
    df = pd.read_csv(comp_csv)
    if df.empty:
        print("model_comparison.csv is empty.")
        return
    best_row = df.iloc[df['AUC'].idxmax()]
    best_model_name = str(best_row['Model'])
    print(f"Best model from artifacts: {best_model_name}")

    # 2) Load best model
    model_path = os.path.join(MODELS_DIR, f"{best_model_name}_best.keras")
    if not os.path.exists(model_path):
        print(f"Best model file not found: {model_path}")
        return
    best_model = tf.keras.models.load_model(model_path)

    # 3) Rebuild test dataset quickly (no augmentation)
    test_ds_local = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_DIR,
        labels='inferred',
        label_mode='binary',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    test_ds_local = test_ds_local.prefetch(buffer_size=AUTOTUNE)

    # 4) Visualize Grad-CAM for all test images
    print("Generating Grad-CAM for all test images from artifacts best model...")
    try:
        visualize_model_interpretability(best_model, test_ds_local, best_model_name, num_images=None)
    except Exception as e:
        print(f"Warning: Grad-CAM generation failed: {e}. Continuing with remaining steps.")

    # 5) External predictions
    external_predictions_csv = os.path.join(REPORTS_DIR, "external_test_images_predictions.csv")
    print("Running external predictions...")
    predict_on_external_images(best_model, external_images_dir, external_predictions_csv)

    # 6) Generate comprehensive report
    # Rebuild results dict from comparison CSV for the report
    results_dict = {}
    for _, row in df.iterrows():
        results_dict[row['Model']] = {
            'accuracy': float(row['Accuracy']),
            'precision': float(row['Precision']),
            'recall': float(row['Recall']),
            'auc': float(row['AUC'])
        }
    print("Creating comprehensive report from artifacts...")
    create_comprehensive_report(results_dict, best_model_name)
    print("Post-processing from artifacts complete.")