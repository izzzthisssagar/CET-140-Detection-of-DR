# Diabetic Retinopathy Detection using Deep Learning

**Student**: Sagar Thapa  
**Institution**: ISMT College, Butwal  
**Student ID**: BI95SS  
**Course**: CET 140 Specialist Project

## 📝 Project Overview
This project implements a deep learning-based system for detecting Diabetic Retinopathy (DR) from retinal fundus images. The system uses transfer learning with state-of-the-art CNN architectures to classify retinal images as either showing signs of diabetic retinopathy or being healthy.

## 🎯 Key Features
- **Multi-model Support**: Implements and compares multiple CNN architectures
- **High Accuracy**: Achieves up to 93.05% AUC score
- **Model Interpretability**: Includes Grad-CAM visualization to understand model decisions
- **Comprehensive Evaluation**: Includes detailed metrics and visualizations
- **Data Analysis**: RGB channel statistics and distribution analysis
- **Easy Deployment**: Models saved in standard formats for easy integration

## 🏗️ Project Structure
```
DR_Project/
├── data/                    # Dataset directories
│   ├── processed/           # Preprocessed images
│   └── raw_images/          # Original dataset
├── models/                  # Trained model files
├── results/                 # Output directory
│   ├── logs/                # Training logs and metrics
│   ├── plots/               # Visualizations and graphs
│   └── reports/             # Generated reports and predictions
├── eye_gradcam_results/     # Grad-CAM visualizations
├── utils/                   # Utility scripts
│   └── image_stats.py      # RGB channel analysis and visualization
├── main.py                 # Main training and evaluation script
├── simple_eye_gradcam.py   # Script for generating Grad-CAM visualizations
├── cleanup.py              # Project cleanup utility
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- CUDA-compatible GPU (recommended)

### Installation
1. Clone the repository:
   ```bash
   git clone [your-repo-url]
   cd DR_Project
   ```

2. Create and activate a virtual environment:
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset from [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection) and place it in the `data/raw_images` directory.

## 🚀 Usage

### Training the Model
To train the model with default settings:
```bash
python main.py
```

### Model Evaluation
The script will automatically evaluate the model on the test set and generate the following in the `results` directory:
- Training history plots
- Confusion matrix
- ROC curve
- Classification report
- Model files

### Making Predictions
To make predictions on new images, place them in the `data/test_images` directory and run:
```bash
python main.py --predict
```

### Generating Grad-CAM Visualizations
To generate Grad-CAM visualizations for your test images:
1. Place your test images in the `C:\Users\LOQ\Desktop\test_images` directory
2. Run the Grad-CAM script:
```bash
python simple_eye_gradcam.py
```
3. The visualizations will be saved in the `eye_gradcam_results` directory, including:
   - Original images
   - Heatmaps showing model focus areas
   - Combined visualizations with heatmaps overlaid on original images

## 📊 Results
Our best performing model achieved the following metrics:

| Metric     | Score |
|------------|-------|
| Accuracy   | 89.34%|
| Precision | 0.89  |
| Recall    | 0.89  |
| F1-Score  | 0.89  |
| AUC-ROC   | 0.9305|

### Model Interpretability with Grad-CAM
Grad-CAM (Gradient-weighted Class Activation Mapping) visualizations help understand which regions of the input image the model focuses on when making predictions. This is particularly useful for medical imaging to ensure the model is making decisions based on clinically relevant features.

Example Grad-CAM output:
![Grad-CAM Example](eye_gradcam_results/example_gradcam.jpg)

*The heatmap shows the model's focus areas, with warmer colors indicating higher importance in the model's decision-making process.*

## 📂 Project Structure
```
DR_Project/
├── data/                    # Dataset directories
│   ├── processed/          # Preprocessed images
│   ├── raw_images/         # Original dataset
│   └── test_images/        # Images for prediction
├── models/                 # Trained model files
├── results/                # Output directory
│   ├── logs/               # Training logs and metrics
│   ├── plots/              # Visualizations and graphs
│   └── reports/            # Generated reports and predictions
├── utils/                  # Utility scripts
│   └── image_stats.py      # RGB channel analysis and visualization
├── main.py                # Main training and evaluation script
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── SUBMISSION.md          # Academic submission document
```

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References
- [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection)
- [Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection)

## 📧 Contact
For any questions or feedback, please open an issue in the repository or contact the project maintainer:

- **Name**: Sagar Thapa  
- **Email**: [sagarthapa2058@gmail.com](mailto:sagarthapa2058@gmail.com)  
- **GitHub**: [izzzthisssagar](https://github.com/izzzthisssagar)  
- **Project Link**: [CET-140-Detection-of-DR](https://github.com/izzzthisssagar/CET-140-Detection-of-DR)
