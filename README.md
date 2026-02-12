# Human Activity Recognition (HAR): A Hybrid Pseudo-Labeling Approach
### This project implements an end-to-end Machine Learning pipeline to identify human activities (Walking and Stair Climbing) using smartphone sensor data (Accelerometer and Gyroscope).
#### The core innovation of this project is the Hybrid Pseudo-Labeling Strategy: using unsupervised clustering (K-Means) cross-referenced with mathematical physics heuristics to generate high-confidence training labels from raw, unlabeled sensor data.
## 🚀 Features
  . Signal Processing: Implementation of Butterworth Low-pass and Median filters to denoise raw 50Hz sensor data.
  . Feature Engineering: Extraction of Time-Domain (Mean, Std Dev, SMA) and Frequency-Domain (FFT) features.
  . Hybrid Labeling: Automated labeling using K-Means and Signal Magnitude Area (SMA) physics equations.
  . Dual-Model Architecture: Comparison between Scikit-Learn (Random Forest) and Deep Learning (PyTorch Neural Network).
  . Deployment: REST API built with Flask for real-time activity prediction.
## 🏗️ Project ArchitectureData Collection: Raw X, Y, Z data from phone sensors.
  . Preprocessing: 50Hz resampling, interpolation, and noise filtering.
  . Feature Extraction: Sliding window (2.56s) with 50% overlap.
  . Pseudo-Labeling:Unsupervised: K-Means (k=2) identifies natural patterns.
  . Heuristic: A mathematical target is set based on $SMA > Threshold$.
  . Training: Supervised learning on the generated pseudo-labels.
  . Serving: Flask API provides Walking vs Stair Climbing classification.
## 🛠️ Installation & Usage1. 
  1. Required libraries pandas numpy matplotlib scipy sklearn torch flask joblib
  2. Data Engineering & LabelingRun the engineering script to process your raw Accelerometer.csv and Gyroscope.csv. This will output labeled_data.csv.Bashpython model_eng.py
  3. Model TrainingTrain the Random Forest and PyTorch Neural Network. This will save scaler.pkl, random_forest_model.pkl, and activity_net.pth.Bashpython model_train.py
  4. Launch the AppStart the Flask server to host your models:Bashpython app.py
## 📊 Research Methodology
   . The Mathematical HeuristicTo validate the clusters, we use the Signal Magnitude Area (SMA) to represent the energy of movement.
   . High energy values combined with high vertical variance (Acc_Z) are mathematically identified as "Stair Climbing," while lower energy cycles are identified as "Walking."
   . Model PerformanceRandom Forest: Typically achieves >90% accuracy on pseudo-labels.
   . Neural Network: A 2-layer Feedforward Network (64 hidden neurons) optimized with Adam.
## 📁 File Structure
   . model_eng.py: Data cleaning, FFT feature extraction, and K-Means labeling.
   . model_train.py: Train/Test split, scaling, and model saving.
   . app.py: Flask API for real-time inference.
   . Accelerometer.csv / Gyroscope.csv: Raw sensor input files.
## 📝 Future WorkIntegration of Turning Detection using Gyroscope Yaw integration.
   .  Implementation of 1D-CNN (Convolutional Neural Network) for raw signal classification.
   .  Development of a Streamlit dashboard for live data visualization.
