# Academic Project Submission

## Student Information
- **Full Name**: Sagar Thapa
- **Institution**: ISMT College, Butwal
- **Student ID**: BI95SS
- **Program**: Bachelor of Science in IT/Computer Science
- **Course**: CET 140 Specialist Project
- **Academic Year**: 2025
- **Supervisor**: [Supervisor's Name]
- **Submission Date**: September 17, 2025

## Project Details
- **Title**: Automated Detection of Diabetic Retinopathy using Deep Learning
- **Project Duration**: January 2025 to September 2025
- **Project Type**: Academic Research & Implementation
- **GitHub Repository**: [https://github.com/izzzthisssagar/CET-140-Detection-of-DR](https://github.com/izzzthisssagar/CET-140-Detection-of-DR)

## Declaration
I hereby declare that this project work titled "Automated Detection of Diabetic Retinopathy using Deep Learning" is my original work and has not been submitted elsewhere for any other degree or diploma. The work presented here is done under the guidance of [Supervisor's Name] at ISMT College, Butwal.

**Signature**: ______________________  
**Date**: __________________________

## Abstract
This project developed an AI system for detecting Diabetic Retinopathy (DR) from retinal fundus images, achieving 93.05% AUC score using deep learning techniques. The system can automatically analyze retinal images and identify signs of diabetic retinopathy with high accuracy, serving as an effective screening tool to assist healthcare professionals in early detection and diagnosis.

## Technical Details
- **Deep Learning Framework**: TensorFlow 2.15.0
- **Base Models**: ResNet50, InceptionV3, EfficientNetB0, EfficientNetB4
- **Best Model**: ResNet50 (AUC: 93.05%)
- **Dataset**: APTOS 2019 Blindness Detection Dataset (3,662 images)
- **Programming Language**: Python 3.8+

## Table of Contents
1. [Introduction](#introduction)
2. [Methodology](#methodology)
3. [Implementation](#implementation)
4. [Results](#results)
5. [Discussion](#discussion)
6. [Future Work](#future-work)
7. [Conclusion](#conclusion)
8. [References](#references)

## 1. Introduction
Diabetic Retinopathy (DR) is a leading cause of blindness among working-age adults worldwide, affecting approximately one-third of people with diabetes. Early detection and treatment can prevent up to 98% of severe vision loss cases. This project addresses the challenge of DR screening by developing an automated deep learning-based system for DR detection.

## 2. Methodology
- **Data Preprocessing**: Image resizing, normalization, and augmentation
- **Model Architecture**: Transfer learning with fine-tuning
- **Training**: 25 epochs with early stopping
- **Evaluation**: Accuracy, Precision, Recall, F1-Score, AUC-ROC

## 3. Implementation
- **Environment**: Python 3.8+, TensorFlow 2.15.0
- **Hardware**: CPU/GPU support
- **Dependencies**: See `requirements.txt`

## 4. Results
- **Best Model**: ResNet50 (AUC: 0.9305)
- **Accuracy**: 89.34%
- **Precision**: 0.89
- **Recall**: 0.89
- **F1-Score**: 0.89

## 5. Discussion
- The model shows strong performance in detecting DR from fundus images
- Grad-CAM visualizations demonstrate the model's focus on clinically relevant features
- Potential for deployment as a screening tool in clinical settings

## 6. Future Work
- Expand to multi-class classification of DR severity
- Develop a web/mobile application for easy access
- Collect and test on more diverse datasets

## 7. Conclusion
The developed system demonstrates the potential of deep learning in medical image analysis, particularly for diabetic retinopathy screening. The high AUC score indicates strong discriminative ability, making it a promising tool for early detection of DR.

## 8. References
1. APTOS 2019 Blindness Detection Competition
2. TensorFlow Documentation
3. Deep Learning for Medical Image Analysis (Academic Papers)

## Appendices
- Source code and documentation available at: [GitHub Repository](https://github.com/izzzthisssagar/CET-140-Detection-of-DR)
- Dataset: [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection)
