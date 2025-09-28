# Blood Group Detection from Fingerprint Images

An AI-powered system that predicts blood groups from fingerprint images using deep learning.

## 🔍 Overview

This project uses a ResNet50-based deep learning model to analyze fingerprint patterns and predict blood groups. The system includes a web interface for easy interaction and real-time predictions.

## 🛠️ Prerequisites

- Python 3.10 or higher
- Windows 10/11
- 8GB RAM minimum
- NVIDIA GPU (optional, but recommended)

## 📦 Installation

1. **Clone or Download Project**
```batch
# Install Git LFS first
git lfs install

# Clone repository with LFS support
git clone https://github.com/Nithin3302/BloodGroup-Detection.git
cd BloodGroup-Detection

# Pull LFS files
git lfs pull
```

2. **Set Up Virtual Environment**
```batch
# Navigate to project directory (adjust path as needed)
cd C:\Users\YourUsername\Projects\bloodgroup_detection

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate
```

3. **Install Dependencies**
```batch
# Ensure you're in the project directory
```C:\Users\YourUsername\Projects\bloodgroup_detection

## 📁 Project Structurey version first
pip install numpy==1.23.5
```
bloodgroup_detection/ependencies
├── data/ll -r requirements.txt
│   └── dataset_blood_group/    # Dataset folder
│       ├── A+/
│       ├── A-/tructure
│       ├── B+/
│       ├── B-/
│       ├── AB+/tion/
│       ├── AB-/
│       ├── O+/_blood_group/    # Dataset folder
│       └── O-/
├── models/ A-/
│   └── blood_group_model.h5    # Trained model
├── src/├── B-/
│   ├── app.py                  # Streamlit interface
│   ├── predict.py              # Prediction logic
│   ├── train.py               # Training script
│   ├── data_processing.py     # Data utilities
│   ├── model.py               # Model architecture
│   └── utils.py               # Helper functions
├── static/
│   └── model_performance.png   # Performance graphse
├── requirements.txt            # Prediction logic
└── README.md.py               # Training script
``` ├── data_processing.py     # Data utilities
│   ├── model.py               # Model architecture
## 🚀 Running the Application  # Helper functions
├── static/
1. **Train Model** (optional if model exists) 
- streamlit run src/app.py    
- ctrl + C in terminal to exit
## 🚀 Running the Application

1. **Train Model** (optional if model exists)
- streamlit run src/app.py    
- ctrl + C in terminal to exit
